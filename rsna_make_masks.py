"""
Step 9a: produce U-Net lung masks for the RSNA data set.

Why a separate script and not `segmentation/make_masks.py`?
  That one is wired to the Kermany folder layout `data/chest_xray/{split}/{class}/`.
  RSNA is flat: `data/rsna/png512/{patientId}.png`, without class folders (the
  label lives in `stage_2_train_labels.csv`).

The three rules that apply here:

  1. THE SEGMENTER GETS ITS OWN PREPROCESSING. The U-Net, a network that decides
     lung or not lung for every pixel, learned on grayscale 256x256 in [0,1]:
     no CLAHE, no per-image standardisation, no ImageNet normalisation. It gets
     exactly that, NOT the classifier transform from
     `rsna_train.build_transforms`.

  2. DOMAIN SHIFT. Trained on Montgomery/Shenzhen (adults, tuberculosis
     screening), applied to RSNA (emergency department, many supine films).
     Hence a QC preview AND an area statistic.

  3. A SUBSET INSTEAD OF EVERYTHING. `--ids-from` takes the existing CAM CSVs
     (~1500 images instead of 26,684).

RAW CACHE, the reason for the second version of this script
-----------------------------------------------------------
The first run gave a lung area of 0.210. Anatomically, 0.30 to 0.40 is what a
frontal chest radiograph should give, and 28.5 % of the bounding box area fell
outside the mask. So the mask undersegments, the same direction as the known
Kermany error, only weaker.

Comparing refinements (convex hull, dilation) would cost a fresh U-Net run per
variant: 15 minutes per setting. That is unnecessary, because only the forward
pass, one run of the images through the network, is expensive; the refinement is
image morphology on a finished binary image. So the RAW U-Net output (256x256,
binary) is cached once as packed bits. 1500 images come to about 12 MB. After
that every further variant costs seconds.

`--refine` picks the refinement without changing the module globals in
`segmentation/mask_refine.py`. That is deliberate: a sweep loop has to compute
several settings in ONE process, and flipping global switches on the way would be
a source of error that silently returns the wrong result.

Output: `data/rsna/masks224/{patientId}.png`, 0/255, 224x224, exactly the grid in
which `pytorch_grad_cam` returns the heatmap. Mask and heatmap therefore line up
without any rescaling.

How to read the two QC numbers:
  Mask area is the fraction of the image the mask covers. Around 0.30 to 0.40 is
  the anatomical expectation for a frontal chest image.
  The area leak AUC asks whether mask area alone separates the classes. Its null
  is 0.5, meaning the area betrays nothing about the class. On Kermany, the
  earlier paediatric data set, it was 0.255, which is what a mask that encodes
  the pathology looks like.

CLI:
  # first run: create masks and raw cache (~15 min CPU)
  python rsna_make_masks.py --ids-from "predictions_rsna/cam_f*_s0.csv" \
      --raw-cache data/rsna/unet_raw256.npz

  # large run: all 22,872 development images, resumable
  #   --flush-every is not a nicety here but a condition. Without the
  #   intermediate saves, 1.4 GiB of raw masks sit in memory until the end, and
  #   an abort in hour three costs three hours. The same call once more resumes
  #   the run where it stopped.
  python rsna_make_masks.py --ids-from qc/dev_ids.csv \
      --masks data/rsna/masks224_dev --raw-cache data/rsna/unet_raw256.npz \
      --refine hull --dilate-px 8 --flush-every 2000 --device directml

  # variant from the cache, without the U-Net (seconds)
  python rsna_make_masks.py --from-cache data/rsna/unet_raw256.npz \
      --refine hull --dilate-px 4 --masks data/rsna/masks224_hull4

  # preview and statistics only
  python rsna_make_masks.py --ids-from "predictions_rsna/cam_f*_s0.csv" --qc-only
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SEG_SIZE = 256          # the size the U-Net learned on
OUT_SIZE = 224          # the size the heatmap comes back in
DEFAULT_CKPT = Path("checkpoints/unet_best.pth")
REFINE_CHOICES = ("none", "default", "hull")


# --------------------------------------------------------------------------
# ID selection  (testable without Torch)
# --------------------------------------------------------------------------

def ids_from_csvs(patterns: list[str]) -> list[str]:
    """Collect patientIds from one or more CSVs, deduplicated and sorted."""
    paths: list[str] = []
    for pat in patterns:
        hits = sorted(globmod.glob(pat))
        if not hits:
            raise FileNotFoundError(f"No match for pattern: {pat}")
        paths.extend(hits)

    ids: list[str] = []
    for p in paths:
        df = pd.read_csv(p)
        if "patientId" not in df.columns:
            raise ValueError(f"{p}: column 'patientId' is missing "
                             f"(present: {list(df.columns)})")
        ids.extend(df["patientId"].astype(str).tolist())
    return sorted(set(ids))


def ids_from_dir(root: Path) -> list[str]:
    return sorted(p.stem for p in Path(root).glob("*.png"))


def balanced_sample(splits_path: Path, n_per_class: int, seed: int = 0) -> list[str]:
    """Draw n images per class from the DEVELOPMENT set (the holdout stays out).

    Needed for every measurement that requires both classes, such as "do the crop
    parameters give away the label?". The existing raw cache is no use for that:
    it comes from the Grad-CAM samples, and Grad-CAM, the heatmap showing where
    the classifier looked, was computed on POSITIVE images only. All 1500 entries
    have Target=1, and an AUC cannot be determined from that.

    The holdout is excluded explicitly. It is reserved for exactly one
    evaluation; touching it for a preparatory measurement would burn that
    evidence silently.
    """
    sp = json.loads(Path(splits_path).read_text())
    holdout = set(sp.get("holdout", []))
    labels = {k: int(v) for k, v in sp["labels"].items() if k not in holdout}

    rng = np.random.default_rng(seed)
    out: list[str] = []
    for cls in (0, 1):
        pool = sorted(k for k, v in labels.items() if v == cls)
        take = min(n_per_class, len(pool))
        out.extend(rng.choice(pool, take, replace=False).tolist())
    return sorted(out)


def pending_jobs(ids: list[str], src: Path, dst: Path,
                 overwrite: bool,
                 cached: set[str] | None = None,
                 ) -> tuple[list[tuple[Path, Path]], int, int]:
    """(pairs still to compute, skipped, missing source images).

    `cached` holds the ids that are already IN THE RAW CACHE. Once a cache is
    kept, an existing mask PNG is not enough as a stop criterion: an aborted run
    may have written the PNG and then never flushed the cache block behind it.
    Without this check that one image falls through both filters on resume and is
    missing from the cache for good, silently, and noticeable only at crop time.
    """
    jobs, skipped, missing = [], 0, 0
    for pid in ids:
        s = Path(src) / f"{pid}.png"
        d = Path(dst) / f"{pid}.png"
        if not s.exists():
            missing += 1
            continue
        done = d.exists() and (cached is None or pid in cached)
        if done and not overwrite:
            skipped += 1
            continue
        jobs.append((s, d))
    return jobs, skipped, missing


# --------------------------------------------------------------------------
# Raw cache: packed bits  (testable without Torch)
# --------------------------------------------------------------------------

def pack_masks(masks: np.ndarray) -> np.ndarray:
    """[n,256,256] bool -> [n, 8192] uint8. Factor of 8 against bool."""
    m = np.asarray(masks, dtype=bool).reshape(len(masks), -1)
    return np.packbits(m, axis=1)


def unpack_masks(packed: np.ndarray, shape=(SEG_SIZE, SEG_SIZE)) -> np.ndarray:
    n = len(packed)
    flat = np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=1)
    return flat[:, :shape[0] * shape[1]].reshape(n, *shape).astype(bool)


def save_raw_cache(path: Path, ids: list[str], masks: np.ndarray) -> None:
    """Merge new entries into an existing cache, which keeps a run resumable.

    Write ATOMICALLY. The cache grows to about 190 MB; overwritten in place, a
    run that dies mid-write loses not the last block but everything. So write
    next to it first, then rename: os.replace is atomic within one file system.

    EVERY np.load here sits in a `with`. A `.npz` is a ZIP archive, and `np.load`
    returns a LAZY `NpzFile` that holds the file open until it is closed. On
    Windows `os.replace` then fails with "WinError 5: access denied", because an
    open file cannot be replaced there. On Linux it would have gone through
    silently, which would only have made the bug later and harder to find.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old_ids: list[str] = []
    old_packed = np.zeros((0, SEG_SIZE * SEG_SIZE // 8), np.uint8)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            old_ids = [str(s) for s in z["ids"]]
            old_packed = np.asarray(z["packed"])      # fetch before closing

    merged: dict[str, np.ndarray] = dict(zip(old_ids, old_packed))
    merged.update(dict(zip(ids, pack_masks(masks))))
    keys = sorted(merged)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, ids=np.array(keys),
                        packed=np.stack([merged[k] for k in keys]))
    os.replace(tmp, path)


def load_raw_cache(path: Path) -> tuple[list[str], np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as z:
        return [str(s) for s in z["ids"]], np.asarray(z["packed"])


def cached_ids(path: Path | None) -> set[str]:
    """Which ids are in the raw cache already? Empty set if there is none."""
    if path is None or not Path(path).exists():
        return set()
    try:
        with np.load(Path(path), allow_pickle=False) as z:
            return {str(s) for s in z["ids"]}
    except Exception as e:                       # half-written file
        print(f"  WARNING: raw cache not readable ({e}). It will be rebuilt.")
        return set()


# --------------------------------------------------------------------------
# Refinement  (testable without Torch)
# --------------------------------------------------------------------------

def refine_variant(raw: np.ndarray, mode: str = "default",
                   dilate_px: int = 0) -> np.ndarray:
    """Refinement with EXPLICIT parameters instead of module globals.

    mode:
      none     the raw U-Net output only (a control: what does the
               post-processing do at all?)
      default  clean up + symmetry fill (the setting used so far)
      hull     additionally a convex hull per lung, which recovers the area a
               consolidation takes away from the segmenter

    dilate_px widens the finished mask elliptically. That is deliberately crude.
    The point is not anatomical accuracy but whether the measured headroom
    disappears once the mask is no longer too small. If it does, the headroom was
    an artefact of the mask.
    """
    if mode not in REFINE_CHOICES:
        raise ValueError(f"mode must be one of {REFINE_CHOICES}, was {mode!r}")

    import cv2

    from segmentation.mask_refine import (_clean, _convex_hull_per_lung,
                                          _symmetry_fill)

    m = np.asarray(raw, dtype=bool)
    if mode != "none":
        m = _clean(m)
        if mode == "hull":
            m = _convex_hull_per_lung(m)
        m = _symmetry_fill(m)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m.astype(np.uint8), k).astype(bool)
    return m


def to_out(mask: np.ndarray, out_size: int = OUT_SIZE) -> np.ndarray:
    """Bring to the classifier grid. NEAREST keeps the mask binary; bilinear
    produced grey values at the borders and distorted the area."""
    import cv2
    m = (np.asarray(mask, dtype=bool).astype(np.uint8) * 255)
    return cv2.resize(m, (out_size, out_size), interpolation=cv2.INTER_NEAREST)


def area_report(areas: np.ndarray) -> dict:
    """Mask area statistics, the QC number next to the visual check.

    A plausible lung covers roughly 0.30 to 0.40 of a frontal chest radiograph.
    The first RSNA run came out at 0.210: too small. Near 0 means the segmenter
    gave up; above about 0.60 it has collected half the image.
    """
    a = np.asarray(areas, dtype=float)
    if a.size == 0:
        return {}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "p05": float(np.percentile(a, 5)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "n_empty": int((a < 0.05).sum()),
        "n_huge": int((a > 0.60).sum()),
        "n_small": int((a < 0.22).sum()),      # below the anatomical expectation
    }


# --------------------------------------------------------------------------
# Torch part
# --------------------------------------------------------------------------

def resolve_device(name: str):
    """Translate a device name into a Torch device, with the same logic as
    training.

    The reason for this detour: `"directml"` is NOT a Torch device name. Torch
    knows cpu/cuda/xpu/..., and `.to("directml")` dies with "Expected one of cpu,
    cuda, ... at start of device string". Only `torch_directml.device()` returns
    the device.

    `rsna_train.pick_device` on purpose and no local copy: this translation
    missing here while training had it is precisely what the bug was. Two
    versions of the same rule drift apart again.

    IDEMPOTENT. An already resolved device is handed back unchanged. Without that
    line a second call falls through every name comparison and silently returns
    the CPU. The run would still run, twenty times slower, and nobody would see
    an error.
    """
    if not isinstance(name, str):
        return name
    from rsna_train import pick_device
    dev, _ = pick_device(name)
    return dev


def _load_unet(ckpt: Path, device: str):
    """Load the U-Net. The checkpoint, meaning the saved trained weights, goes to
    the CPU ALWAYS and is copied into the model from there.

    `map_location=<DirectML device>` dies with
    "TypeError: '>=' not supported between instances of 'torch.device' and 'int'",
    because Torch passes the device object on to `torch_directml.device()`, which
    expects an integer index there. The error looks like a broken checkpoint and
    is not one.

    `weights_only=True` also suppresses the FutureWarning; the fallback catches
    old Torch versions without that argument only.
    """
    import torch
    from segmentation.unet import UNet

    model = UNet(base_ch=32).to(resolve_device(device))
    try:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    except TypeError:                       # older Torch versions
        state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _preprocess(path: Path):
    """Exactly as in segmenter training: grayscale, 256x256 bilinear, [0,1]."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    img = Image.open(path).convert("L")
    img = TF.resize(img, [SEG_SIZE, SEG_SIZE],
                    interpolation=InterpolationMode.BILINEAR)
    return TF.to_tensor(img)                     # [1,256,256] in [0,1]


def generate(jobs: list[tuple[Path, Path]], ckpt: Path, device: str, batch: int,
             mode: str, dilate_px: int,
             raw_cache: Path | None,
             flush_every: int = 0) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Compute masks and store them. Returns (areas, ids, raw 256x256 masks).

    `flush_every > 0` writes the raw cache out every N images and empties the
    buffer. Two reasons, both binding at 22,872 images:

      Memory. One raw image is 256x256 bool = 64 KiB, all of them together
      1.4 GiB, and `np.stack` at the end briefly doubles that.

      Loss. Without intermediate saves an abort in hour three costs all three
      hours. With them it costs at most N images, and
      `pending_jobs(..., cached=...)` picks the run up again exactly there.

    At flush_every=0 (the default, small runs) the buffer is returned and main()
    saves once. At flush_every>0 everything is in the cache already on exit, so
    an EMPTY buffer comes back and main() does not write a second time.
    """
    import torch

    dev = resolve_device(device)          # translate once, then use everywhere
    model = _load_unet(ckpt, dev)
    areas, ids, raws = [], [], []
    n_flushed = 0
    total = len(jobs)
    t0 = time.time()

    def flush() -> int:
        """Merge the buffer into the cache and empty it. Returns the count."""
        nonlocal ids, raws
        if raw_cache is None or not ids:
            return 0
        n = len(ids)
        save_raw_cache(raw_cache, ids, np.stack(raws))
        ids, raws = [], []
        return n

    with torch.no_grad():
        for i in range(0, total, batch):
            chunk = jobs[i:i + batch]
            x = torch.stack([_preprocess(s) for s, _ in chunk]).to(dev)
            pred = (torch.sigmoid(model(x)) > 0.5).cpu().numpy()[:, 0]
            for (src, out_path), p in zip(chunk, pred):
                out = to_out(refine_variant(p, mode, dilate_px))
                Image.fromarray(out, mode="L").save(out_path)
                areas.append(float((out > 127).mean()))
                if raw_cache is not None:
                    ids.append(src.stem)
                    raws.append(p)
            done = min(i + batch, total)
            if done % (batch * 10) == 0 or done == total:
                el = time.time() - t0
                rest = el / done * (total - done)
                print(f"    masks {done}/{total}  "
                      f"({el / 60:.0f} min elapsed, ~{rest / 60:.0f} min left)",
                      flush=True)
            if flush_every and len(ids) >= flush_every:
                n_flushed += flush()
                print(f"    intermediate save done ({n_flushed} in this run)",
                      flush=True)

    if flush_every:
        n_flushed += flush()
        return np.asarray(areas), [], np.zeros((0, SEG_SIZE, SEG_SIZE), bool)

    raw_arr = np.stack(raws) if raws else np.zeros((0, SEG_SIZE, SEG_SIZE), bool)
    return np.asarray(areas), ids, raw_arr


def from_cache(cache: Path, dst: Path, mode: str, dilate_px: int,
               only: list[str] | None) -> np.ndarray:
    """Re-refine masks from the raw cache, without Torch, in seconds."""
    ids, packed = load_raw_cache(cache)
    keep = set(only) if only else None
    Path(dst).mkdir(parents=True, exist_ok=True)

    areas = []
    for i, pid in enumerate(ids):
        if keep is not None and pid not in keep:
            continue
        raw = unpack_masks(packed[i:i + 1])[0]
        out = to_out(refine_variant(raw, mode, dilate_px))
        Image.fromarray(out, mode="L").save(Path(dst) / f"{pid}.png")
        areas.append(float((out > 127).mean()))
    return np.asarray(areas)


def measured_areas(ids: list[str], dst: Path) -> np.ndarray:
    out = []
    for pid in ids:
        p = Path(dst) / f"{pid}.png"
        if p.exists():
            out.append(float((np.array(Image.open(p)) > 127).mean()))
    return np.asarray(out)


# --------------------------------------------------------------------------
# QC
# --------------------------------------------------------------------------

def qc_preview(ids: list[str], src: Path, dst: Path, csv_dir: Path,
               out_png: Path, n_per_class: int = 4, seed: int = 0) -> None:
    """Preview with the mask overlaid, split by label.

    Split by label, because that is where the known error sits: on Kermany,
    pneumonic lungs were UNDERsegmented (the consolidation does not look like
    lung to the segmenter). If that comes back, it has to be visible before the
    mask decides anything.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = pd.read_csv(Path(csv_dir) / "stage_2_train_labels.csv")
    lab = (labels.groupby("patientId")["Target"].max()).to_dict()

    have = [i for i in ids if (Path(dst) / f"{i}.png").exists()]
    rng = np.random.default_rng(seed)
    rows: list[tuple[str, str]] = []
    for name, want in (("Target=0", 0), ("Target=1", 1)):
        pool = [i for i in have if lab.get(i) == want]
        if not pool:
            continue
        pick = rng.choice(pool, min(n_per_class, len(pool)), replace=False)
        rows.extend((name, str(p)) for p in pick)

    if not rows:
        print("  QC: no masks found to display.")
        return

    fig, axes = plt.subplots(len(rows), 2, figsize=(6, 3 * len(rows)),
                             squeeze=False)
    for r, (name, pid) in enumerate(rows):
        img = np.array(Image.open(Path(src) / f"{pid}.png").convert("L")
                       .resize((OUT_SIZE, OUT_SIZE)))
        m = np.array(Image.open(Path(dst) / f"{pid}.png")) > 127
        axes[r][0].imshow(img, cmap="gray")
        axes[r][0].set_title(f"{name}  {pid[:8]}", fontsize=8)
        axes[r][1].imshow(img, cmap="gray")
        axes[r][1].imshow(m, cmap="Reds", alpha=0.35)
        axes[r][1].set_title(f"mask  area {m.mean():.3f}", fontsize=8)
        for c in (0, 1):
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=90)
    plt.close(fig)
    print(f"  QC preview: {out_png}")


def area_leak_check(ids: list[str], dst: Path, csv_dir: Path) -> float | None:
    """Does mask area on its own already give away the class?

    Exactly the leak that was found on Kermany (lung_area AUC ~0.255). It has to
    be known BEFORE any crop experiment: a crop onto a mask whose size gives away
    the class builds the shortcut into the image.

    None if only one class is present. The CAM subset has no negatives, because
    Grad-CAM was measured on positive images only.
    """
    from sklearn.metrics import roc_auc_score

    labels = pd.read_csv(Path(csv_dir) / "stage_2_train_labels.csv")
    lab = (labels.groupby("patientId")["Target"].max()).to_dict()

    y, a = [], []
    for pid in ids:
        p = Path(dst) / f"{pid}.png"
        if pid in lab and p.exists():
            y.append(int(lab[pid]))
            a.append(float((np.array(Image.open(p)) > 127).mean()))
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, a))


def print_area(rep: dict) -> None:
    if not rep:
        return
    print(f"\nMask area (fraction of the image), n={rep['n']}:")
    print(f"  mean {rep['mean']:.3f} +- {rep['sd']:.3f} | P05 {rep['p05']:.3f} | "
          f"median {rep['median']:.3f} | P95 {rep['p95']:.3f}")
    print(f"  empty (<0.05): {rep['n_empty']} | huge (>0.60): {rep['n_huge']} | "
          f"below anatomical expectation (<0.22): {rep['n_small']}")
    if rep["mean"] < 0.26:
        print("  -> TOO SMALL. Anatomically ~0.30 to 0.40 is expected. A mask")
        print("     that is too small produces 'peak outside the lung' by itself.")
        print("     Remedy: --refine hull and/or --dilate-px.")


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, default=Path("data/rsna/png512"))
    p.add_argument("--masks", type=Path, default=Path("data/rsna/masks224"))
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--ids-from", nargs="+", default=None,
                   help="CSV(s) with a patientId column; glob allowed")
    p.add_argument("--all", action="store_true",
                   help="all PNGs under --images (26,684 of them, ~1 h)")
    p.add_argument("--balanced-sample", type=int, default=None,
                   metavar="N",
                   help="N images PER CLASS from the development set (holdout "
                        "excluded), for measurements that need both classes")
    p.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    p.add_argument("--refine", choices=REFINE_CHOICES, default="default",
                   help="none = raw | default = clean+symmetry | "
                        "hull = plus convex hull")
    p.add_argument("--dilate-px", type=int, default=0,
                   help="widen the mask by N pixels (in the 256 grid)")
    p.add_argument("--raw-cache", type=Path, default=None,
                   help="cache the raw U-Net output here (later variants "
                        "then cost nothing)")
    p.add_argument("--from-cache", type=Path, default=None,
                   help="re-refine masks from the raw cache, without the U-Net")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--qc-only", action="store_true")
    p.add_argument("--qc-out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--flush-every", type=int, default=0, metavar="N",
                   help="save the raw cache every N images, which makes the run "
                        "resumable. Required for large runs, otherwise 1.4 GiB "
                        "of raw masks sit in memory until the end.")
    args = p.parse_args(argv)

    if not (args.all or args.ids_from or args.from_cache or args.balanced_sample):
        p.error("give one of --ids-from, --all, --balanced-sample or --from-cache")
    qc_out = args.qc_out or Path("qc") / f"rsna_mask_qc_{args.masks.name}.png"

    args.masks.mkdir(parents=True, exist_ok=True)

    # ---- path A: from the raw cache, without Torch --------------------------
    if args.from_cache:
        only = None
        if args.ids_from:
            only = ids_from_csvs(args.ids_from)
        elif args.balanced_sample:
            only = balanced_sample(args.splits, args.balanced_sample, args.seed)
        elif args.all:
            only = ids_from_dir(args.images)
        print(f"From raw cache: {args.from_cache} | refine={args.refine} "
              f"dilate={args.dilate_px} -> {args.masks}")
        areas = from_cache(args.from_cache, args.masks, args.refine,
                           args.dilate_px, only)
        ids = ids_from_dir(args.masks)
        print(f"  {len(areas)} masks written (no U-Net needed).")

    # ---- path B: run the U-Net ---------------------------------------------
    else:
        if args.balanced_sample:
            ids = balanced_sample(args.splits, args.balanced_sample, args.seed)
        elif args.all:
            ids = ids_from_dir(args.images)
        else:
            ids = ids_from_csvs(args.ids_from)
        print(f"Images in the selection: {len(ids)}")
        have = cached_ids(args.raw_cache) if args.raw_cache is not None else None
        if have is not None:
            print(f"  already in the raw cache: {len(have)}")
        jobs, skipped, missing = pending_jobs(ids, args.images, args.masks,
                                              args.overwrite, cached=have)
        print(f"  to compute: {len(jobs)} | present: {skipped} | "
              f"source missing: {missing}")
        if missing:
            print("  CAUTION: missing source images point to a wrong --images "
                  "path or to an incomplete conversion.")

        if args.qc_only:
            print("  --qc-only: nothing will be computed.")
        elif jobs:
            if not args.ckpt.exists():
                print(f"ERROR: checkpoint missing: {args.ckpt}")
                return 2
            print(f"  device: {args.device} | checkpoint: {args.ckpt} | "
                  f"refine={args.refine} dilate={args.dilate_px}")
            if args.flush_every:
                print(f"  raw cache is saved every {args.flush_every} images. "
                      f"From then on an abort costs at most that many images.")
            _, cids, raws = generate(jobs, args.ckpt, args.device, args.batch,
                                     args.refine, args.dilate_px, args.raw_cache,
                                     flush_every=args.flush_every)
            if args.raw_cache is not None and len(cids):
                save_raw_cache(args.raw_cache, cids, raws)
            if args.raw_cache is not None:
                n_now = len(cached_ids(args.raw_cache))
                print(f"  raw cache: {args.raw_cache} ({n_now} entries in total)")
                print("  -> further variants now with --from-cache, no U-Net.")
        else:
            print("  nothing to do (all masks are present).")

    # ---- QC: numbers first, picture afterwards ------------------------------
    print_area(area_report(measured_areas(ids, args.masks)))

    auc = area_leak_check(ids, args.masks, args.csv)
    if auc is None:
        print("\nArea leak: not determinable (only one class in the selection).")
    else:
        print(f"\nArea leak: AUC(mask area -> Target) = {auc:.3f}")
        print("  0.5 = the area gives nothing away. On Kermany it was 0.255.")

    qc_preview(ids, args.images, args.masks, args.csv, qc_out, seed=args.seed)
    print("\nDone. Look at the preview first, then rsna_mask_sweep.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
