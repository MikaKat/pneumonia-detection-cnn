"""
Step 9e: precompute the lung crop.

The crop and its three decisions
--------------------------------
The crop is the squared bounding rectangle of the lung mask plus a margin.
Three decisions sit in that sentence, each resting on an earlier measurement.

  RECTANGLE, NOT A PIXEL-EXACT MASK. The pixel-exact variant is rejected, its
  three proposed mechanisms refuted one by one. The rectangle is fixed by the
  lung apices and the costophrenic angles, and the U-Net segments those
  reliably even when it loses a consolidation in mid-lung. That is where the
  pixel-exact mask became dangerous: it cut the pathology away with the rest.

  SQUARE, NOT WITH INDEPENDENT SIDE LENGTHS. `rsna_crop_geometry.py` measured
  the framing against the projection at AUC 0.745, the aspect ratio alone at
  0.623. AUC is the probability that a randomly drawn AP film scores above a
  randomly drawn PA one; 0.5 would mean the framing says nothing about the
  projection. Scaling a non-square rectangle to 224x224 distorts the image by
  an amount that depends on the aspect ratio, so the aspect ratio survives the
  crop as a distortion signature. The channel is not closed, only re-encoded.

  SHIFT RATHER THAN SHRINK. A square that hangs partly outside the image is
  pushed inwards, not made smaller. Shrinking would tie the crop size to the
  position in the image, and the position is itself a projection proxy. That
  is the same error through the back door.

The adaptive window and its measured failure
--------------------------------------------
The default above derives the side length from the lung bounding box, and that
is what the first crop training run used. It failed on the pre-registered
endpoint: AUC(score -> ViewPosition) ROSE in all five folds, mean +0.027, while
the stratified AUC did not move (+0.003). The mechanism was measured afterwards
in `qc/crop_varianten_tabelle.csv`: the window side length alone predicts AP/PA
at AUC 0.685, against 0.714 for the whole framing. An adaptive crop therefore
takes almost the entire channel it is meant to close and writes it back into
the image as a global zoom factor, at a median 1.197x for AP against 1.117x for
PA. Texture frequency is easier for a convolutional net to read than framing,
so the crop made the confounder worse rather than better.

`--fixed-side`: the only neutral crop form
------------------------------------------
The same measurement pointed at the remedy. With the side length held CONSTANT
and only the position taken from the mask, the geometry channel falls from
0.714 to 0.554, and 0.554 is position alone. Retained box area at side 0.80 is
0.996; below 0.75 the box retention breaks down. 0.80 is therefore the value
with a safety margin, 0.75 the aggressive case.

Two consequences that are easy to get wrong:

  A MISSING MASK MUST NOT MEAN AN UNCROPPED IMAGE. With the adaptive window,
  falling back to the full image is merely one more size among many. With a
  fixed side it would be the ONLY image at a different zoom, which is exactly
  the per-image size difference the fixed side exists to remove. Under
  `--fixed-side` the fallback is therefore a CENTRED window of the same side
  length, and `ok=False` still records it.

  `--shift-y` IS CONSTANT ON PURPOSE. The lost box area sits at the bottom
  (0.65 % against 0.09 % at the top at side 0.75): the mask underestimates the
  diaphragm, so the costophrenic angles are cut first. A constant downward
  offset recovers part of that and can carry no per-image information, because
  it is identical for every image. An offset derived from the mask could not
  make that promise.

The smoke test for a fixed side is `side_ptp` in the report, the spread max
minus min. It has to be EXACTLY 0.0. Anything else means some path still
derives the side length from the image, and then the run measures the adaptive
window again under a new name.

Until 07.08.2026 this sentence said `side_sd`, and that was wrong in a way that
only bites when everything else is right: a standard deviation over identical
values is 1.11e-16 rather than 0.0, so the confirming line never printed while
the printed number read 0.000000. See `crop_report`.

PRECOMPUTE, NOT AT RUNTIME. Cropping during training would mean loading the
mask for every image on every pass over the data. Under DirectML the DataLoader
runs with `--workers 0`, so that would be the bottleneck.

The boxes have to come along
----------------------------
The pneumonia bounding boxes live in the 1024 DICOM grid of the ORIGINAL
IMAGE. After the crop they point at the wrong place. Grad-CAM marks the image
regions that carried the score, so the marking can be held against the
annotated box. Held against boxes that were not transformed along, it yields a
hit rate that looks plausible and means nothing.
`test_rsna_train.test_box_geometry` covers that case.

The script therefore writes its own `stage_2_train_labels.csv` into the crop
folder, with coordinates in the 1024 grid OF THE CROP. `rsna_train.load_boxes`
then works unchanged and only needs `--csv data/rsna/crop512`.

Reading the output
------------------
The first number to read is how much annotated pathology the crop cuts away.
If a meaningful share of the box area is gone, the matter is settled whatever
the AUC says: the model can no longer see what it is being asked about. That
number therefore stands at the top of the report instead of somewhere below it.

CLI:
  # crop the development set (masks have to exist already)
  python rsna_make_crops.py --ids-from qc/dev_ids.csv \
      --raw-cache data/rsna/unet_raw256.npz --out data/rsna/crop512

  # later, for the one-off holdout evaluation
  python rsna_make_crops.py --ids-from qc/holdout_ids.csv \
      --raw-cache data/rsna/unet_raw256.npz --out data/rsna/crop512

  # the neutral variant: constant size, position from the mask
  python rsna_make_crops.py --ids-from qc/dev_ids.csv \
      --raw-cache data/rsna/unet_raw256.npz \
      --out data/rsna/crop512_fix080 --fixed-side 0.80 --shift-y 0.03 \
      --params-out predictions_rsna/crop_params_fix080.csv

  # then train:
  python rsna_train.py --fold 0 --images data/rsna/crop512 \
      --csv data/rsna/crop512
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BOX_SPACE = 1024          # the grid the boxes use in stage_2_train_labels.csv


# --------------------------------------------------------------------------
# Geometry  (pure arithmetic, no Torch and no cv2, so that it stays testable)
# --------------------------------------------------------------------------

def mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Bounding rectangle of a mask, normalised to [0,1].

    Returns (top, left, bottom, right), or None for an empty mask. The
    normalisation keeps the number off the 256 mask grid, since it is applied
    to the 512 PNG.
    """
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.where(m)
    if ys.size == 0:
        return None
    H, W = m.shape
    return (ys.min() / H, xs.min() / W, (ys.max() + 1) / H, (xs.max() + 1) / W)


def centred_crop(side: float, shift_y: float = 0.0
                 ) -> tuple[float, float, float]:
    """The fallback window when no mask is available and the side is fixed.

    Centred, same side length as every other image. See the module header for
    why the uncropped full image is the wrong fallback in that case.
    """
    side = min(max(side, 0.0), 1.0)
    top = min(max((1.0 - side) / 2 + shift_y, 0.0), 1.0 - side)
    return top, (1.0 - side) / 2, side


def square_crop(bbox: tuple[float, float, float, float],
                pad: float = 0.05,
                fixed_side: float | None = None,
                shift_y: float = 0.0) -> tuple[float, float, float]:
    """Turn the rectangle into a square crop window.

    Everything in [0,1]; the source image is square (512x512 from
    `rsna_prepare.py`), so square in these units is square in pixels as well.

    Returns (top, left, side).

    The margin is taken relative to the RESPECTIVE side (pad*h at the top and
    bottom, pad*w to the left and right) and only THEN squared off. Squaring
    first and adding one common margin afterwards would tie the margin width to
    the longer side, and on supine images the longer side is a different one
    than on upright ones.

    With `fixed_side` the side length is NOT derived from the rectangle. Only
    the centre is, and `pad` then has no effect at all, because a margin around
    a window of constant size cannot change its size. `shift_y` moves the
    window down by a constant fraction of the image; positive is downwards.
    """
    t, l, b, r = bbox
    h, w = b - t, r - l
    t, b = t - pad * h, b + pad * h
    l, r = l - pad * w, r + pad * w

    if fixed_side is None:
        side = min(max(b - t, r - l), 1.0)      # never larger than the image
    else:
        side = min(max(fixed_side, 0.0), 1.0)
    cy, cx = (t + b) / 2 + shift_y, (l + r) / 2
    top, left = cy - side / 2, cx - side / 2

    # Shift inwards instead of shrinking; see the module header.
    top = min(max(top, 0.0), 1.0 - side)
    left = min(max(left, 0.0), 1.0 - side)
    return top, left, side


def transform_boxes(boxes: list[tuple[float, float, float, float]],
                    crop: tuple[float, float, float],
                    box_space: float = BOX_SPACE
                    ) -> tuple[list[tuple[float, float, float, float]], float]:
    """Convert boxes into the grid of the crop and clip them at its margin.

    Input and output are each (x, y, w, h) in the `box_space` grid, the output
    relative to the CROP, which again counts as `box_space` wide. The format of
    `stage_2_train_labels.csv` stays unchanged and `rsna_train.load_boxes`
    works without modification.

    Second return value: the fraction of the original box area that has been
    RETAINED. 1.0 = nothing cut off. That is the number the crop can fail on.
    """
    top, left, side = crop
    kept, area_in, area_out = [], 0.0, 0.0
    for x, y, w, h in boxes:
        # Carry both areas in THE SAME unit: normalised to the original image.
        # `area_out` is converted back with side*side below; leaving `area_in`
        # in box_space units would be a silent factor of 1024^2 and would make
        # the retained fraction meaninglessly small.
        area_in += (w / box_space) * (h / box_space)
        # in [0,1] of the original image
        x0, y0 = x / box_space, y / box_space
        x1, y1 = (x + w) / box_space, (y + h) / box_space
        # in [0,1] of the crop
        nx0, ny0 = (x0 - left) / side, (y0 - top) / side
        nx1, ny1 = (x1 - left) / side, (y1 - top) / side
        # clip
        nx0, ny0 = max(nx0, 0.0), max(ny0, 0.0)
        nx1, ny1 = min(nx1, 1.0), min(ny1, 1.0)
        if nx1 <= nx0 or ny1 <= ny0:
            continue                                    # fell out completely
        # back into the box_space grid, now the one of the crop
        bx, by = nx0 * box_space, ny0 * box_space
        bw, bh = (nx1 - nx0) * box_space, (ny1 - ny0) * box_space
        kept.append((bx, by, bw, bh))
        area_out += (nx1 - nx0) * (ny1 - ny0) * side * side
    frac = (area_out / area_in) if area_in > 0 else float("nan")
    return kept, frac


def crop_report(rows: list[dict]) -> dict:
    """The numbers to read before training starts."""
    if not rows:
        return {}
    d = pd.DataFrame(rows)
    out = {
        "n": len(d),
        "n_leer": int((~d["ok"]).sum()),
        "side_median": float(d["side"].median()),
        # The smoke test for --fixed-side, and it is `side_ptp`, not `side_sd`.
        #
        # `side_sd` runs over mean and squares, so on 22872 identical values of
        # 0.800 it does not come out as 0.0 but as 1.11e-16. Printed with six
        # decimals that reads "0.000000" and looks perfect, while
        # `side_sd == 0.0` is False and the confirming line in `print_report`
        # never appears. A watchdog keyed to that line would abort a completely
        # correct run: the same class of mistake as the culture bug in the
        # phase 6 runner, only the other way round.
        #
        # The spread max minus min carries no such noise. On identical values
        # it is exactly 0.0, which is what the module header has meant all
        # along. `side_sd` stays in the report unchanged, nothing is dropped.
        "side_sd": float(d["side"].std(ddof=0)),
        "side_ptp": float(d["side"].max() - d["side"].min()),
        "zoom_median": float((1.0 / d["side"]).median()),
        "flaeche_median": float((d["side"] ** 2).median()),
    }
    mit = d.dropna(subset=["box_frac"])
    if len(mit):
        out.update({
            "n_mit_box": len(mit),
            "box_erhalt_mittel": float(mit["box_frac"].mean()),
            "box_erhalt_min": float(mit["box_frac"].min()),
            "n_box_unter_90": int((mit["box_frac"] < 0.90).sum()),
            "n_box_ganz_weg": int((mit["box_frac"] <= 0.0).sum()),
            "anteil_ganz_weg": float((mit["box_frac"] <= 0.0).mean()),
        })
    return out


def print_report(rep: dict) -> None:
    if not rep:
        print("  (nothing to report)")
        return
    print(f"\n  Images                      {rep['n']}")
    print(f"  without a usable mask       {rep['n_leer']}  "
          f"(taken over uncropped)")
    print(f"  crop side length            {rep['side_median']:.3f} of the image "
          f"(Median)")
    print(f"  area                        {rep['flaeche_median']:.3f}  "
          f"| linear zoom {rep['zoom_median']:.2f}x")
    print(f"  spread of the side length   {rep['side_ptp']:.6f}  "
          f"(max minus min; with --fixed-side this has to be 0.000000)")
    print(f"    standard deviation        {rep['side_sd']:.3e}  "
          f"(floating point noise on a constant, see crop_report)")
    if rep["side_ptp"] == 0.0:
        print("  -> CONSTANT WINDOW SIZE, the zoom carries no per-image "
              "information.")
    if "n_mit_box" in rep:
        print(f"\n  --- What the crop takes away from the pathology ---")
        print(f"  images with a box           {rep['n_mit_box']}")
        print(f"  retained box area           {rep['box_erhalt_mittel']:.4f} "
              f"(mean)   minimum {rep['box_erhalt_min']:.3f}")
        print(f"  images below 90 % retained  {rep['n_box_unter_90']}")
        print(f"  boxes lost entirely         {rep['n_box_ganz_weg']}  "
              f"({rep['anteil_ganz_weg'] * 100:.2f} %)")
        # The thresholds are FRACTIONS, not counts. A fixed number such as
        # "more than zero" is bound to trigger on 5000 images and would weigh a
        # single case like a systematic error.
        if rep["box_erhalt_mittel"] < 0.95 or rep["anteil_ganz_weg"] > 0.005:
            print("\n  WARNING: the crop cuts annotated pathology away.")
            print("  The AUC question is then secondary: the model can no")
            print("  longer see what it is asked about. Raise pad first.")
        else:
            print("  -> the crop leaves the annotated pathology standing.")
            if rep["n_box_ganz_weg"]:
                print(f"     ({rep['n_box_ganz_weg']} isolated case(s) lose the box")
                print(f"     entirely, below the 0.5 % limit, so no pattern.)")


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

def load_label_table(csv_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(csv_dir) / "stage_2_train_labels.csv")


def boxes_by_id(df: pd.DataFrame) -> dict[str, list[tuple[float, float, float, float]]]:
    """Like `rsna_train.load_boxes`, but from an already loaded frame."""
    d = df[df["Target"] == 1].dropna(subset=["x", "y", "width", "height"])
    out: dict[str, list] = {}
    for pid, x, y, w, h in d[["patientId", "x", "y", "width", "height"]].values:
        out.setdefault(str(pid), []).append((float(x), float(y),
                                             float(w), float(h)))
    return out


def run(ids: list[str], images: Path, out_dir: Path, cache: Path,
        refine: str, dilate_px: int, pad: float, size: int,
        csv_dir: Path, overwrite: bool,
        fixed_side: float | None = None, shift_y: float = 0.0) -> list[dict]:
    """Crop each image, write it out and record what the crop did to it."""
    import cv2
    from PIL import Image

    from rsna_make_masks import load_raw_cache, refine_variant, unpack_masks

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_ids, packed = load_raw_cache(cache)
    index = {pid: i for i, pid in enumerate(cache_ids)}

    label_df = load_label_table(csv_dir)
    boxes = boxes_by_id(label_df)

    rows: list[dict] = []
    ohne_bild, ohne_maske = 0, 0
    t0 = time.time()
    for k, pid in enumerate(ids, 1):
        src = images / f"{pid}.png"
        dst = out_dir / f"{pid}.png"
        if not src.exists():
            ohne_bild += 1
            continue

        # If the mask is missing, the image is taken over UNCROPPED and noted
        # as ok=False. Simply leaving it out would be worse: the crop folder
        # would then be incomplete, and training would abort only hours later
        # on a missing file. In exchange the count appears in the report; a
        # silent substitution would itself be an error.
        if pid in index:
            raw = unpack_masks(packed[index[pid]:index[pid] + 1])[0]
            bb = mask_bbox(refine_variant(raw, refine, dilate_px))
        else:
            ohne_maske += 1
            bb = None
        ok = bb is not None
        if ok:
            crop = square_crop(bb, pad, fixed_side, shift_y)
        elif fixed_side is None:
            crop = (0.0, 0.0, 1.0)
        else:
            # Not the full image: that would be the only differently sized
            # window in the set. See the module header.
            crop = centred_crop(fixed_side, shift_y)

        if overwrite or not dst.exists():
            img = np.array(Image.open(src).convert("L"))
            H, W = img.shape
            if H != W:
                raise ValueError(f"{pid}: source image is not square "
                                 f"({H}x{W}); square_crop presupposes it.")
            top, left, side = crop
            y0, x0 = int(round(top * H)), int(round(left * W))
            s = max(int(round(side * H)), 1)
            y0 = min(y0, H - s); x0 = min(x0, W - s)
            patch = img[y0:y0 + s, x0:x0 + s]
            # INTER_AREA when shrinking, INTER_LINEAR when enlarging. AREA
            # averages and avoids the aliasing that would otherwise travel into
            # the model as a texture difference.
            interp = cv2.INTER_AREA if s > size else cv2.INTER_LINEAR
            Image.fromarray(cv2.resize(patch, (size, size),
                                       interpolation=interp)).save(dst)

        neu, frac = ([], float("nan"))
        if pid in boxes:
            neu, frac = transform_boxes(boxes[pid], crop)
        rows.append({"patientId": pid, "ok": ok,
                     "top": crop[0], "left": crop[1], "side": crop[2],
                     # A constant column, and it earns its place. `top` mixes
                     # the mask centre with the offset, so once the run is over
                     # the offset cannot be recovered from the sum. Without
                     # this column, `--shift-y` is the one parameter of the
                     # crop that nothing downstream can check.
                     "shift_y": float(shift_y),
                     "box_frac": frac if pid in boxes else np.nan,
                     "n_boxes_vorher": len(boxes.get(pid, [])),
                     "n_boxes_nachher": len(neu)})
        for bx, by, bw, bh in neu:
            rows[-1].setdefault("_neu", []).append((bx, by, bw, bh))

        if k % 2000 == 0 or k == len(ids):
            el = time.time() - t0
            print(f"    crop {k}/{len(ids)}  ({el / 60:.1f} min)", flush=True)

    if ohne_bild:
        print(f"  WARNING: {ohne_bild} source images missing, skipped.")
    if ohne_maske:
        print(f"  WARNING: {ohne_maske} images without a mask in the raw cache, "
              f"taken over uncropped.")
    return rows


def write_labels(rows: list[dict], csv_dir: Path, out_dir: Path) -> Path:
    """New `stage_2_train_labels.csv` in the crop folder.

    The layout follows the original: negatives as one row with NaN
    coordinates, positives one row per box. That is what keeps
    `rsna_train.load_boxes` usable unchanged, which is the point of the file.
    """
    orig = load_label_table(csv_dir)
    target = dict(zip(orig["patientId"].astype(str), orig["Target"]))

    out = []
    for r in rows:
        pid = r["patientId"]
        t = int(target.get(pid, 0))
        neu = r.get("_neu", [])
        if not neu:
            out.append({"patientId": pid, "x": np.nan, "y": np.nan,
                        "width": np.nan, "height": np.nan, "Target": t})
            continue
        for bx, by, bw, bh in neu:
            out.append({"patientId": pid, "x": bx, "y": by,
                        "width": bw, "height": bh, "Target": t})

    path = out_dir / "stage_2_train_labels.csv"
    pd.DataFrame(out).to_csv(path, index=False)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, default=Path("data/rsna/png512"))
    p.add_argument("--out", type=Path, default=Path("data/rsna/crop512"))
    p.add_argument("--raw-cache", type=Path,
                   default=Path("data/rsna/unet_raw256.npz"))
    p.add_argument("--csv", type=Path, default=Path("data/rsna"),
                   help="folder holding the original stage_2_train_labels.csv")
    p.add_argument("--ids-from", nargs="+", required=True,
                   help="CSV(s) with a patientId column; globs allowed")
    p.add_argument("--refine", default="hull",
                   help="mask refinement, as measured in rsna_crop_geometry.py")
    p.add_argument("--dilate-px", type=int, default=8)
    p.add_argument("--pad", type=float, default=0.05)
    p.add_argument("--fixed-side", type=float, default=None,
                   help="hold the window size CONSTANT at this fraction of "
                        "the image and take only the position from the mask. "
                        "0.80 is the measured value with a safety margin, "
                        "0.75 the aggressive one. Without this flag the "
                        "adaptive window is used, which raised the primary "
                        "endpoint in all five folds; see the module header.")
    p.add_argument("--shift-y", type=float, default=0.0,
                   help="constant downward offset of the window as a fraction "
                        "of the image; 0.03 recovers part of the box area "
                        "lost at the diaphragm. Constant for every image on "
                        "purpose, so it cannot encode anything per image.")
    p.add_argument("--size", type=int, default=512,
                   help="edge length of the output; 512 matches png512, so "
                        "nothing else in rsna_train.py has to change")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--params-out", type=Path,
                   default=Path("predictions_rsna/crop_params.csv"))
    args = p.parse_args(argv)

    if args.fixed_side is not None and not 0.0 < args.fixed_side <= 1.0:
        print(f"ERROR: --fixed-side has to lie in (0, 1], not "
              f"{args.fixed_side}.")
        return 2
    # Overwriting a parameter file silently is the same class of error as the
    # overwritten checkpoints: the file then belongs to a different crop than
    # the images that were evaluated with it.
    if args.params_out.exists() and not args.overwrite:
        print(f"ERROR: {args.params_out} already exists.")
        print("  Choose a different --params-out for a new crop variant, or")
        print("  pass --overwrite if this run really is meant to replace it.")
        return 2

    from rsna_make_masks import ids_from_csvs
    ids = ids_from_csvs(args.ids_from)
    print(f"Images in the selection: {len(ids)}")
    if args.fixed_side is None:
        print("  window: ADAPTIVE (size from the mask). This is the variant "
              "that raised the primary endpoint by 0.027.")
    else:
        print(f"  window: FIXED at {args.fixed_side:.3f} of the image, "
              f"downward offset {args.shift_y:.3f}, position from the mask.")
    if not args.raw_cache.exists():
        print(f"ERROR: raw cache missing: {args.raw_cache}")
        print("  Run rsna_make_masks.py first.")
        return 2

    rows = run(ids, args.images, args.out, args.raw_cache, args.refine,
               args.dilate_px, args.pad, args.size, args.csv, args.overwrite,
               args.fixed_side, args.shift_y)
    if not rows:
        print("ERROR: nothing was cropped.")
        return 2

    labels = write_labels(rows, args.csv, args.out)
    print(f"\n  boxes in crop coordinates -> {labels}")

    args.params_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if k != "_neu"} for r in rows]
                 ).to_csv(args.params_out, index=False)
    print(f"  crop parameters           -> {args.params_out}")

    print_report(crop_report(rows))
    print(f"\nDone. Training:  python rsna_train.py --fold N "
          f"--images {args.out} --csv {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
