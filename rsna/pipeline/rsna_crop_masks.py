"""Step 9f: carry the LUNG MASKS into the coordinate frame of the crop.

Why this file exists
--------------------
`rsna_make_crops.py` carries two things into the crop: the pixels and the
pneumonia boxes. It does not carry the lung masks, and until phase 7 that was
fine, because nothing downstream needed them in the crop frame.

Phase 7 needs them. Two measurements are taken INSIDE the lung mask and against
a LOCATION PRIOR:

  * the smoke test of `rsna_kopf_auswertung.py`, head field against the
    location prior, point AUC within the lung mask;
  * endpoint B of the roadmap, the localisation measurement.

The head field of a phase 7 arm lives in the frame of the CROP. `masks224_dev`
and `predictions_lokalisation/prior_f*.npy` live in the frame of the WHOLE
image. Holding one against the other compares two maps from different worlds,
and it fails in the dangerous direction: the location prior is built from box
positions in the uncropped frame, so against a cropped head field it is too
weak a null line, the head looks better than it is, and the smoke test passes
too easily. A gate that only ever opens is not a gate.

So the masks get the SAME window the images got, out of the same parameter
file, and the prior is then rebuilt from the crop boxes:

    python rsna_crop_masks.py --params predictions_rsna/crop_params_fix080.csv \
        --masks data/rsna/masks224_dev --out data/rsna/masks224_dev_fix080

    python rsna/befunde/rsna_lokalisation.py tor \
        --csv data/rsna/crop512_fix080 \
        --masks data/rsna/masks224_dev_fix080 \
        --out-dir predictions_lokalisation_fix080

Three decisions
---------------
  NEAREST, NOT AREA. `load_lung` reads the mask back as `m > 127`. Averaging
  interpolation would put grey values on every lung border and then the
  threshold, not the segmentation, would decide where the lung ends. The image
  itself is resized with AREA for the opposite reason: there the grey values
  are the content.

  THE WINDOW COMES FROM THE PARAMETER FILE, NOT FROM THE MASK. Recomputing it
  here from the mask would be a second implementation of `square_crop`, and two
  implementations of one window are one window too many: the day they drift
  apart, the masks are cut differently from the images and nothing says so.
  `--params` is therefore required and has no default.

  THE MASK GRID IS 224, THE IMAGE GRID IS 512. The window is in [0,1] of the
  image, so it applies unchanged; only the pixel arithmetic differs. The
  rounding error is NOT half a pixel: each edge collects 0.5/224 from the mask
  grid and 0.5/512 from the image grid, and the integer side length adds its
  own (410/512 = 0.80078 against 179/224 = 0.79911). Measured over the range of
  offsets that actually occur, an edge moves by up to about one mask pixel,
  0.5 percent of the frame. That is still below the resolution of everything
  this mask is used for, but it is twice what half a pixel would suggest.

The number to read is the retained lung area. If the crop cut lung away, then
"point AUC within the lung mask" would quietly change its meaning between the
two arms, and the comparison would stop being paired in the thing that matters.

CLI:
  python rsna_crop_masks.py --params predictions_rsna/crop_params_fix080.csv \
      --masks data/rsna/masks224_dev --out data/rsna/masks224_dev_fix080
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def crop_one(mask: np.ndarray, top: float, left: float, side: float,
             size: int) -> np.ndarray:
    """Apply one window to one mask and scale it back to `size`.

    Pure arithmetic on an array, so it is testable without files. The pixel
    arithmetic is deliberately the same shape as in `rsna_make_crops.run`:
    round the corner, take the side as an integer, and push the window inside
    the image instead of shrinking it.
    """
    import cv2

    H, W = mask.shape
    if H != W:
        raise ValueError(f"mask is {H}x{W}, square expected")
    y0, x0 = int(round(top * H)), int(round(left * W))
    s = max(int(round(side * H)), 1)
    y0 = min(max(y0, 0), H - s)
    x0 = min(max(x0, 0), W - s)
    patch = mask[y0:y0 + s, x0:x0 + s]
    # NEAREST in both directions. The mask is a decision, not a picture.
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_NEAREST)


def report(rows: list[dict]) -> dict:
    if not rows:
        return {}
    d = pd.DataFrame(rows)
    return {
        "n": len(d),
        # Two different denominators, and that is the point: the lung fills
        # more of the CROP than it did of the whole image, and that difference
        # is the zoom, not a gain. `erhalt` is the number without the zoom in
        # it, see the comment at the call site.
        "anteil_ganzes_bild": float(d["vorher"].mean()),
        "anteil_im_zuschnitt": float(d["im_zuschnitt"].mean()),
        "erhalt_mittel": float(d["erhalt"].mean()),
        "erhalt_min": float(d["erhalt"].min()),
        "n_unter_95": int((d["erhalt"] < 0.95).sum()),
        "n_leer": int((d["im_zuschnitt"] <= 0.0).sum()),
    }


def print_report(rep: dict) -> None:
    if not rep:
        print("  (nothing to report)")
        return
    print(f"\n  masks                       {rep['n']}")
    print(f"  lung, share of the whole image   "
          f"{rep['anteil_ganzes_bild']:.4f}")
    print(f"  lung, share of the crop          "
          f"{rep['anteil_im_zuschnitt']:.4f}  (larger: that is the zoom)")
    print(f"  retained lung area          {rep['erhalt_mittel']:.4f} (mean)   "
          f"minimum {rep['erhalt_min']:.3f}")
    print(f"  masks below 95 % retained   {rep['n_unter_95']}")
    print(f"  masks empty after the crop  {rep['n_leer']}")
    if rep["n_leer"] or rep["erhalt_mittel"] < 0.95:
        print("\n  WARNING: the crop cuts lung away. Every measurement taken")
        print("  INSIDE the lung mask then means something different in the")
        print("  two arms, and the paired comparison loses its pairing.")
    else:
        print("  -> the crop leaves the lung standing; measurements inside the")
        print("     mask stay comparable between the arms.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", type=Path, required=True,
                   help="crop_params CSV written by rsna_make_crops.py; the "
                        "window comes from here and is NOT recomputed")
    p.add_argument("--masks", type=Path, default=Path("data/rsna/masks224_dev"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", type=int, default=224,
                   help="edge length of the output; 224 is what load_lung "
                        "expects and it raises on anything else")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    from PIL import Image

    if not args.params.exists():
        print(f"ERROR: {args.params} missing. Run rsna_make_crops.py first.")
        return 2
    par = pd.read_csv(args.params)
    for spalte in ("patientId", "top", "left", "side"):
        if spalte not in par.columns:
            print(f"ERROR: {args.params} has no column '{spalte}'.")
            return 2

    spanne = float(par["side"].max() - par["side"].min())
    print(f"Windows from {args.params}: {len(par)}")
    print(f"  side length {par['side'].median():.3f} (median), "
          f"spread {spanne:.6f}")
    if spanne == 0.0:
        print("  -> a FIXED side. Every mask is cut at the same size.")
    else:
        print("  -> an ADAPTIVE side. That is the variant which raised the")
        print("     primary endpoint by 0.027; check that this is intended.")

    # An output folder that already holds masks is refused, not quietly kept.
    # Without this, `if not dst.exists()` would leave the old files on disk
    # while the report below describes the NEW window: run this twice with a
    # different `--shift-y` and the folder holds one crop while the printed
    # retention describes another, with nothing saying so. Same class of
    # mistake as an overwritten `--params-out` in `rsna_make_crops.py`, and
    # that one is refused there for the same reason.
    args.out.mkdir(parents=True, exist_ok=True)
    schon_da = sum(1 for _ in args.out.glob("*.png"))
    if schon_da and not args.overwrite:
        print(f"ERROR: {args.out} already holds {schon_da} PNGs.")
        print("  They may come from a different window. Either delete the")
        print("  folder, choose a different --out, or pass --overwrite if this")
        print("  run really is meant to replace them.")
        return 2

    rows, fehlt = [], 0
    t0 = time.time()
    for k, (pid, top, left, side) in enumerate(
            par[["patientId", "top", "left", "side"]].itertuples(index=False),
            1):
        src = args.masks / f"{pid}.png"
        dst = args.out / f"{pid}.png"
        if not src.exists():
            fehlt += 1
            continue
        m = np.array(Image.open(src).convert("L"))
        neu = crop_one(m, float(top), float(left), float(side), args.size)
        if args.overwrite or not dst.exists():
            Image.fromarray(neu, mode="L").save(dst)
        vorher = float((m > 127).mean())
        im_zuschnitt = float((neu > 127).mean())
        # Retention needs BOTH areas in the same unit, and the unit is the
        # fraction of the ORIGINAL image. The crop covers side*side of it, so
        # the fraction measured in the crop is scaled back by that factor.
        # Dividing the two frame fractions directly would report a gain of
        # 1/side^2 on every single mask, and that is the zoom, not a retained
        # area. With side 0.80 it would read 1.56 everywhere and look like the
        # crop had grown lungs.
        nachher = im_zuschnitt * float(side) * float(side)
        rows.append({"patientId": pid, "vorher": vorher,
                     "im_zuschnitt": im_zuschnitt,
                     "erhalt": (nachher / vorher) if vorher > 0 else np.nan})
        if k % 2000 == 0 or k == len(par):
            print(f"    mask {k}/{len(par)}  ({(time.time() - t0) / 60:.1f} min)",
                  flush=True)

    if fehlt:
        print(f"  WARNING: {fehlt} masks missing in {args.masks}, skipped.")
    if not rows:
        print("ERROR: nothing was cropped.")
        return 2

    d = pd.DataFrame(rows)
    d = d.dropna(subset=["erhalt"])
    print_report(report(d.to_dict("records")))
    print(f"\nDone: {args.out}")
    print("Next, the location prior in the same frame:")
    print(f"  python rsna\\befunde\\rsna_lokalisation.py tor --csv <cropfolder> "
          f"--masks {args.out} --out-dir <priorfolder>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
