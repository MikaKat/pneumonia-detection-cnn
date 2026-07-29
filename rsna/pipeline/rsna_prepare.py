"""
DICOM -> PNG. Run once; afterwards the dataset is treated exactly like Kermany.

Why convert at all: `pydicom.dcmread` plus JPEG decoding costs a multiple of a
PNG load per image, and that cost comes back in every epoch (one epoch = one
pass of the whole training set through the network). With 26 684 images and
the DirectML constraint `--workers 0` (see READMEforMe; otherwise Windows
spawns a fresh Torch import per worker), decoding would be the bottleneck and
the GPU would sit waiting for it.

Deliberate decisions:

* 512x512, not 1024. Training runs at 224. 512 still leaves room for later
  384 experiments or a lung crop without touching 26 684 DICOMs again, and it
  costs ~7 GB instead of ~25 GB. With 60 GB of free disk space, that is the
  relevant size.
* No CLAHE, no normalisation, no crop. The PNG is a faithful copy, only
  smaller. Every preprocessing decision belongs in the dataset transform, where
  it can be switched per run and is therefore comparable. On Kermany, CLAHE was
  baked into the conversion, which made preprocessing and dataset impossible to
  tell apart afterwards.
* Only `stage_2_train_images`. The test folder has no labels and is worthless
  here.
* Threads, not processes. The work is predominantly file I/O and C decoding,
  and both release the GIL. Under Windows, processes would hit the same spawn
  problem as the DataLoader workers.

Interpreting the output:
  The preflight block prints transfer syntax, image size, bit depth and pixel
  value range of the first DICOM. If decoding fails there, a JPEG plugin is
  missing and the run stops before any file has been written. The run then
  reports how many DICOMs exist, how many are already converted and how many
  remain, so a repeated call on a complete raw cache does nothing. At the end
  it prints elapsed time, the number of PNGs written and the total size on
  disk; the expected result is one PNG per DICOM. A non-empty FAILED list means
  the raw cache is incomplete. The script exits with status 1, and those files
  have to be resolved before the cache is used for training.

CLI:
  python rsna_prepare.py --dicom data/rsna/stage_2_train_images \
      --out data/rsna/png512 --size 512 --workers 8
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image


def preflight(files: list[Path]) -> None:
    """Decode a single image as a trial BEFORE touching 26 684 files.

    RSNA DICOMs are JPEG-compressed. pydicom only decodes that with a matching
    plugin; if the plugin is missing, the error surfaces in the middle of the
    run and looks like a data problem. Better to fail immediately and say so
    clearly.
    """
    import pydicom

    ds = pydicom.dcmread(str(files[0]))
    ts = ds.file_meta.TransferSyntaxUID
    print(f"Preflight on {files[0].name}")
    print(f"  TransferSyntax : {ts.name if hasattr(ts, 'name') else ts}")
    print(f"  Size           : {ds.Rows}x{ds.Columns}, "
          f"{ds.BitsStored} bit, {ds.PhotometricInterpretation}")
    try:
        arr = ds.pixel_array
    except Exception as e:                                   # noqa: BLE001
        print(f"\n  Decoding failed: {type(e).__name__}: {e}")
        print("  Missing JPEG plugin. Install it in the active venv:")
        print("      pip install pylibjpeg pylibjpeg-libjpeg")
        print("  (not while a training run is active, see READMEforMe/Hardware)")
        sys.exit(1)
    print(f"  Pixels         : dtype {arr.dtype}, "
          f"min {arr.min()}, max {arr.max()}\n")


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Bring the array to 8 bit, without any hidden contrast adjustment.

    RSNA is 8 bit MONOCHROME2 throughout, so the first branch always applies.
    The second is an honest emergency brake: it scales linearly across the full
    value range. This is NOT windowing. If it ever did apply, a deliberate
    windowing belongs here rather than a silent autoscale, because an
    image-dependent scaling is exactly the kind of global contrast
    normalisation that became the shortcut on Kermany.
    """
    if arr.dtype == np.uint8:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, np.uint8)
    return (((arr.astype(np.float32) - lo) / (hi - lo)) * 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dicom", type=Path, default=Path("data/rsna/stage_2_train_images"))
    p.add_argument("--out", type=Path, default=Path("data/rsna/png512"))
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Sanity run. Note: the file order is NOT "
                        "random (see READMEforMe phase 4).")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    files = sorted(args.dicom.glob("*.dcm"))
    if not files:
        raise SystemExit(f"No .dcm files under {args.dicom}")
    if args.limit:
        files = files[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    preflight(files)

    todo = files if args.overwrite else [
        f for f in files if not (args.out / f"{f.stem}.png").exists()]
    print(f"{len(files)} DICOMs, {len(files) - len(todo)} already converted, "
          f"{len(todo)} to do -> {args.out} ({args.size}x{args.size})")
    if not todo:
        return

    import pydicom

    done, failed = 0, []
    lock = Lock()
    t0 = time.time()

    def convert(f: Path) -> None:
        nonlocal done
        try:
            ds = pydicom.dcmread(str(f))
            img = Image.fromarray(to_uint8(ds.pixel_array)).convert("L")
            if img.size != (args.size, args.size):
                img = img.resize((args.size, args.size), Image.LANCZOS)
            # write to a temporary file first, then rename: otherwise an abort
            # in the middle of writing leaves half a PNG behind, which the next
            # run counts as "already finished"
            tmp = args.out / f"{f.stem}.png.tmp"
            img.save(tmp, "PNG", optimize=False)
            tmp.replace(args.out / f"{f.stem}.png")
        except Exception as e:                               # noqa: BLE001
            with lock:
                failed.append((f.name, f"{type(e).__name__}: {e}"))
        with lock:
            done += 1
            if done % 2000 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(todo)}  {el:5.0f}s  "
                      f"(remaining ~{el / done * (len(todo) - done):.0f}s)")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(convert, todo))

    total_mb = sum(f.stat().st_size for f in args.out.glob("*.png")) / 1e6
    print(f"\ndone in {time.time() - t0:.0f}s | {len(todo) - len(failed)} "
          f"written | {total_mb:.0f} MB in {args.out}")
    if failed:
        print(f"\n{len(failed)} FAILED:")
        for name, err in failed[:10]:
            print(f"  {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
