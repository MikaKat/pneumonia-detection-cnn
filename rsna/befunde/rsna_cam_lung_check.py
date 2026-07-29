"""
Step 9b: does the heat map point into the lung? Answered from the existing
checkpoints, without retraining.

The question that decides 11.5 hours of compute
-----------------------------------------------
Grad-CAM reads a trained classifier backwards and returns, for one radiograph,
a coarse heat map of the regions that drove its pneumonia score. The maximum
of that map falls inside an annotated bounding box in 53.9 % of cases against
a chance value of 11.7 %, a factor of 4.6. The *mass* of the map lies in the
boxes only to 19.2 %. The map points in roughly the right direction and is
diffuse at the same time. The next step on offer is a crop to the lung
(`crop`), so that the model never sees image border, shoulders and abdomen in
the first place, at a cost of 11.5 h over five folds. The checkpoints, the
weights already written out by the finished training runs, allow a check of
whether such a crop CAN change anything before that time is spent.

EVERY NUMBER NEEDS ITS DENOMINATOR
----------------------------------
An outside share means nothing until the share a random point would produce
stands beside it. For "outside" that chance value is 1 - lung area, here
0.792, and the measured lift over it is +0.014 +- 0.100 at t = 0.32 across
five folds. Nothing. On the hits the same comparison comes out differently:
maximum in the lung at 0.858 against a chance value of 0.212.

The model is bimodal. Either it finds the pathology, with the maximum in the
lung and in the box, or its maximum is noise with respect to the anatomy. A
systematic "looks at ribs, diaphragm, image border" does not exist, consistent
with the corner ablation, where taking the corners out of the image and
rescoring moved the result by -0.0001.

Every outside number therefore carries its baseline beside it, and the verdict
hangs on the lift, not on the raw value. The headroom is subject to the same
rule. A purely random heat map gains +0.264 from being restricted to the lung,
simply because space is taken away from it. An observed gain has to be held
against that value, not against zero.

TWO CONTROLS BEFORE ANY FINDING
-------------------------------
1. `box_in_lung`: which share of the bounding box lies inside the mask? The
   mask comes from a U-Net, a network that labels every pixel lung or not
   lung. On Kermany, pneumonia lungs were undersegmented, because a
   consolidation does not look like lung to it. The pathology then falls out
   of the mask and "maximum outside the lung" arises by itself. Circular.
2. `lung_area`: anatomically 0.30-0.40 is to be expected, and the measured
   value is 0.210. A mask that is too small drives every outside number AND
   the headroom upwards, without the model having anything to do with it.

HEAT CACHE
----------
`--cache-heat` stores the heat maps as float16 (~12 MB per fold). The
comparison of different mask variants then becomes a table lookup taking
seconds instead of 20 minutes of CAM computation per variant, which is what
`rsna_mask_sweep.py` uses.

Interpreting the output
-----------------------
Every rate is printed beside its chance baseline, and only the lift between
the two carries meaning. All figures are means +- SD over the folds, the five
patient splits the model was trained and validated on, and the differences
carry a paired t across folds (df = k-1, so |t| > 2.78 is the 5 % bound with
five folds). A single-fold number decides nothing here, because fold
difficulty varies more than the effects being measured.

The report is to be read in this order:

  1. Mask controls. If `box_in_lung` or `lung_area` fails its threshold, there
     is no reading at all: a mask that is too small or displaced produces the
     outside finding and the headroom on its own, and any interpretation would
     be circular. Repair the mask first (`rsna_mask_sweep.py`).
  2. Failures whose maximum lies outside the lung, against the chance value
     1 - lung area. Only a lift with |t| > 2.78 shows that the model
     systematically looks at ribs, diaphragm or image border.
  3. Headroom, stated as two lifts, each against its own null: free
     (hit rate - box area) and restricted (hit rate inside the lung minus
     (box AND lung) / lung). Their difference is immune to the ceiling
     objection, because both terms are distances to a matching null.

The decision. A significant outside lift TOGETHER WITH a difference of the two
lifts above the purely geometric gain argues FOR the crop run: the mechanism a
crop addresses is then present, and the model is spending its attention on
ribs, diaphragm or image border, which a crop removes from view. An outside
share indistinguishable from chance, or a headroom that stays within plain
area arithmetic, argues AGAINST it. The crop cannot help a model that already
looks inside the lung and only in the wrong place, and the 11.5 h are saved.

CLI:
  python rsna_cam_lung_check.py --folds 0 1 2 3 4 --n 120 --cache-heat
  python rsna_cam_lung_check.py --folds 0 --n 300 --masks data/rsna/masks224_hull4
"""

from __future__ import annotations

import _repo_path  # noqa: F401  (setzt sys.path fuer die Nachbarordner)

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

LUNG_AREA_MIN = 0.26        # below this the mask is anatomically implausibly small
BOX_IN_LUNG_MIN = 0.60      # below this the mask cuts the pathology away


# --------------------------------------------------------------------------
# Geometry and evaluation  (torch-free, so it stays testable and usable in the sweep)
# --------------------------------------------------------------------------

def box_mask(boxes, size: int, box_space: int) -> np.ndarray:
    """Bounding boxes from the 1024 px DICOM grid into the model grid (size).

    Returns a boolean map that is True inside any annotated box. The boxes
    carry the original DICOM resolution, and the model works on a smaller
    grid, so without this rescaling every hit rate would be measured against
    the wrong geometry.
    """
    s = size / box_space
    m = np.zeros((size, size), bool)
    for bx, by, bw, bh in boxes:
        y0, y1 = max(int(by * s), 0), int((by + bh) * s)
        x0, x1 = max(int(bx * s), 0), int((bx + bw) * s)
        m[y0:y1, x0:x1] = True
    return m


def analyse_one(heat: np.ndarray, box: np.ndarray, lung: np.ndarray) -> dict | None:
    """All metrics for one image.

    `peak_in_box_lungrestricted` is the optimism ceiling: the maximum is
    searched for only INSIDE the lung. A model trained on crops would do
    something else, since it learns different weights from the start. This is
    the upper bound on what taking away the field of view alone could deliver.

    `null_restricted` is the matching denominator, the hit rate of a RANDOM
    point inside the lung, that is (box intersected with lung) / lung. Without
    that number every gain looks like model merit when it can be plain area
    arithmetic.

    The peak coordinates are recorded as well, so that a different mask can be
    scored afterwards without a new CAM computation.
    """
    heat = np.clip(np.asarray(heat, dtype=float), 0, None)
    total = heat.sum()
    if total <= 0:
        return None

    yx = np.unravel_index(int(np.argmax(heat)), heat.shape)

    if lung.any():
        masked = np.where(lung, heat, -1.0)
        yx_l = np.unravel_index(int(np.argmax(masked)), masked.shape)
        peak_lr = bool(box[yx_l])
    else:
        yx_l = yx
        peak_lr = bool(box[yx])           # no mask -> no restriction

    box_a = float(box.mean())
    lung_a = float(lung.mean())
    inter = float((box & lung).mean())
    return {
        "peak_y": int(yx[0]), "peak_x": int(yx[1]),
        "peak_lung_y": int(yx_l[0]), "peak_lung_x": int(yx_l[1]),
        "peak_in_box": bool(box[yx]),
        "peak_in_lung": bool(lung[yx]),
        "peak_in_box_lungrestricted": peak_lr,
        "mass_in_box": float(heat[box].sum() / total),
        "mass_in_lung": float(heat[lung].sum() / total),
        "box_area": box_a,
        "lung_area": lung_a,
        "box_in_lung": float((box & lung).sum() / box.sum()) if box.any() else np.nan,
        # Denominators for the headroom: chance hit free, and restricted to the lung
        "null_free": box_a,
        "null_restricted": float(min(inter / lung_a, 1.0)) if lung_a > 0 else box_a,
    }


def summarise(df: pd.DataFrame) -> dict:
    """Fold summary. Every rate stands beside its baseline.

    Takes the per-image table of one fold and returns a single row: the two
    mask controls, the hit rates each with their chance value, and the two
    lifts whose difference is the headroom. No rate is emitted alone. A raw
    value on its own cannot be falsified; a lift against its null can.
    """
    d = df.dropna(subset=["peak_in_box"])
    miss = d[~d["peak_in_box"].astype(bool)]
    hit = d[d["peak_in_box"].astype(bool)]

    out = {
        "n": int(len(d)),
        "box_in_lung": float(d["box_in_lung"].mean()),
        "lung_area": float(d["lung_area"].mean()),
        "box_area": float(d["box_area"].mean()),
        "peak_in_box": float(d["peak_in_box"].mean()),
        "peak_in_lung": float(d["peak_in_lung"].mean()),
        "peak_in_lung_lift": float(d["peak_in_lung"].mean() - d["lung_area"].mean()),
        "mass_in_lung": float(d["mass_in_lung"].mean()),
        "mass_in_box": float(d["mass_in_box"].mean()),
        "n_miss": int(len(miss)),
        "hit_in_lung": float(hit["peak_in_lung"].mean()) if len(hit) else np.nan,
        "hit_in_lung_null": float(hit["lung_area"].mean()) if len(hit) else np.nan,
    }

    # The fraction at issue, raw value AND denominator.
    if len(miss):
        out["miss_outside_lung"] = float((~miss["peak_in_lung"].astype(bool)).mean())
        out["miss_outside_null"] = float(1 - miss["lung_area"].mean())
    else:
        out["miss_outside_lung"] = np.nan
        out["miss_outside_null"] = np.nan
    out["miss_outside_lift"] = out["miss_outside_lung"] - out["miss_outside_null"]

    out["peak_in_box_lungrestricted"] = float(d["peak_in_box_lungrestricted"].mean())
    out["crop_headroom"] = out["peak_in_box_lungrestricted"] - out["peak_in_box"]
    # What the same restriction would deliver for a random heat map:
    out["null_free"] = float(d["null_free"].mean())
    out["null_restricted"] = float(d["null_restricted"].mean())
    out["headroom_null"] = out["null_restricted"] - out["null_free"]

    # The same quantity in the reading that survives cross-examination.
    #
    # Comparing gains (model +0.080 against chance +0.264) is open to attack:
    # the model starts at 0.530, chance at 0.117, so the model has less room
    # above it. The objection "ceiling effect" would be justified.
    #
    # Hence two LIFTS instead, each against its own baseline:
    #   free        hit rate - box area
    #   restricted  hit rate(lung only) - (box AND lung)/lung
    # Both are distances to the matching null and therefore directly
    # comparable. Algebraically the difference is identical to
    # headroom_vs_null. In this form it is immune to the ceiling objection.
    out["lift_free"] = out["peak_in_box"] - out["null_free"]
    out["lift_restricted"] = out["peak_in_box_lungrestricted"] - out["null_restricted"]
    out["lift_delta"] = out["lift_restricted"] - out["lift_free"]
    out["headroom_vs_null"] = out["lift_delta"]
    return out


def cv_mean(rows: list[dict], key: str) -> tuple[float, float]:
    """Mean +- SD over the folds. A single-fold number is worth nothing here,
    because fold difficulty varies more than the effects being measured, so
    absolute values are reported only as a cross-validation mean."""
    v = np.array([r[key] for r in rows if key in r and not np.isnan(r[key])],
                 dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def paired_t(rows: list[dict], key: str) -> float:
    """Paired t of a difference column across the folds. df = k-1, so with five
    folds |t| > 2.78 is the 5 % bound. Deliberately without scipy: a single
    t statistic does not justify an additional dependency."""
    v = np.array([r[key] for r in rows if key in r and not np.isnan(r[key])],
                 dtype=float)
    if v.size < 2:
        return float("nan")
    sd = v.std(ddof=1)
    if sd == 0:
        return float("inf") if v.mean() != 0 else 0.0
    return float(v.mean() / (sd / np.sqrt(v.size)))


# --------------------------------------------------------------------------
# Torch part
# --------------------------------------------------------------------------

def load_lung(masks: Path, pid: str, size: int) -> np.ndarray | None:
    p = Path(masks) / f"{pid}.png"
    if not p.exists():
        return None
    m = np.array(Image.open(p).convert("L"))
    if m.shape != (size, size):
        # No silent resize. The mask is produced at model resolution on
        # purpose, so a size that does not fit means an assumption is wrong.
        raise ValueError(f"{p}: mask is {m.shape}, expected ({size}, {size}). "
                         f"Run rsna_make_masks.py with a matching OUT_SIZE.")
    return m > 127


def run_fold(fold: int, args) -> pd.DataFrame:
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

    from rsna_train import BOX_SPACE, build_transforms, load_boxes, make_model

    ckpt = Path(f"checkpoints/rsna_f{fold}_s{args.seed}.pth")
    cam_csv = Path(args.pred_dir) / f"cam_f{fold}_s{args.seed}.csv"
    if not ckpt.exists():
        print(f"  Fold {fold}: checkpoint missing ({ckpt}), skipped.")
        return pd.DataFrame()
    if not cam_csv.exists():
        print(f"  Fold {fold}: CAM CSV missing ({cam_csv}), skipped.")
        return pd.DataFrame()

    # The same images as in the reported number, not drawn afresh.
    stored = pd.read_csv(cam_csv)
    ids = stored["patientId"].astype(str).tolist()
    if args.n and args.n < len(ids):
        rng = np.random.default_rng(args.seed)
        ids = list(rng.choice(ids, args.n, replace=False))

    boxes = load_boxes(args.csv)
    model = make_model(torch.device("cpu"))
    try:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    tf = build_transforms(args.size, False)

    rows, no_mask = [], 0
    heat_ids, heats = [], []
    for j, pid in enumerate(ids, 1):
        lung = load_lung(args.masks, pid, args.size)
        if lung is None:
            no_mask += 1
            continue
        img = Image.open(Path(args.images) / f"{pid}.png").convert("L")
        x = tf(img).unsqueeze(0)
        heat = np.clip(cam(input_tensor=x,
                           targets=[BinaryClassifierOutputTarget(1)])[0], 0, None)
        r = analyse_one(heat, box_mask(boxes.get(pid, []), args.size, BOX_SPACE),
                        lung)
        if r is None:
            continue
        r["patientId"] = pid
        rows.append(r)
        if args.cache_heat:
            heat_ids.append(pid)
            heats.append(heat.astype(np.float16))
        if j % 50 == 0:
            print(f"    Fold {fold}: {j}/{len(ids)}")

    if no_mask:
        print(f"  Fold {fold}: {no_mask} images without a mask skipped. "
              f"Run rsna_make_masks.py for these IDs.")

    if args.cache_heat and heats:
        out = Path(args.pred_dir) / f"cam_heat_f{fold}_s{args.seed}.npz"
        np.savez_compressed(out, ids=np.array(heat_ids), heat=np.stack(heats))
        print(f"  Fold {fold}: heat cache {out.name} "
              f"({len(heats)} maps, {out.stat().st_size / 1e6:.0f} MB)")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["fold"] = fold

    # Cross-check against the stored number. Same images, same checkpoint and
    # the same transform, so the hit rate MUST come out the same again.
    merged = df.merge(stored[["patientId", "hit"]], on="patientId", how="inner")
    if len(merged):
        agree = float((merged["peak_in_box"] == merged["hit"]).mean())
        print(f"  Fold {fold}: reproduction of the stored hit rate "
              f"{agree:.3f} over {len(merged)} images"
              + ("" if agree > 0.98 else "   <-- WARNING, should be ~1.000"))
    return df


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _line(label: str, m: float, s: float, signed: bool = False) -> str:
    fmt = "%+.3f" if signed else "%.3f"
    return f"  {label:<46} " + (fmt % m) + f" +- {s:.3f}"


def report(per_fold: list[dict]) -> None:
    if not per_fold:
        print("No results.")
        return

    print("\n" + "=" * 76)
    print("CONTROLS FIRST: is the mask usable as a yardstick?")
    print("=" * 76)
    bil, bil_s = cv_mean(per_fold, "box_in_lung")
    la, la_s = cv_mean(per_fold, "lung_area")
    print(_line("Bounding box inside the mask", bil, bil_s))
    print(_line("Lung area (anatomically ~0.30-0.40)", la, la_s))

    mask_ok = True
    if bil < BOX_IN_LUNG_MIN:
        mask_ok = False
        print(f"  -> box_in_lung < {BOX_IN_LUNG_MIN}: the mask cuts the "
              f"pathology away.")
    if la < LUNG_AREA_MIN:
        mask_ok = False
        print(f"  -> lung area < {LUNG_AREA_MIN}: the mask is too small. It")
        print("     produces 'maximum outside the lung' and headroom by itself.")
        print("     Remedy: rsna_make_masks.py --refine hull --dilate-px N")
    if mask_ok:
        print("  -> both controls passed.")

    print("\n" + "=" * 76)
    print("FINDING: every rate beside its baseline")
    print("=" * 76)
    for k, kn, label in [
        ("peak_in_box", "box_area", "Maximum inside a box"),
        ("peak_in_lung", "lung_area", "Maximum inside the lung"),
        ("hit_in_lung", "hit_in_lung_null", "  of those: hits, max inside the lung"),
        ("mass_in_box", "box_area", "Heat-map mass inside the boxes"),
        ("mass_in_lung", "lung_area", "Heat-map mass inside the lung"),
    ]:
        m, s = cv_mean(per_fold, k)
        n, _ = cv_mean(per_fold, kn)
        print(f"  {label:<40} {m:.3f} +- {s:.3f}   chance {n:.3f}   "
              f"lift {m - n:+.3f}")

    print("\n" + "-" * 76)
    print("THE DECIDING QUESTION: on failures, does the model look OUT of the lung?")
    print("-" * 76)
    mo, mo_s = cv_mean(per_fold, "miss_outside_lung")
    mn, _ = cv_mean(per_fold, "miss_outside_null")
    ml, ml_s = cv_mean(per_fold, "miss_outside_lift")
    t_miss = paired_t(per_fold, "miss_outside_lift")
    print(f"  Failure, maximum outside the lung          {mo:.3f} +- {mo_s:.3f}")
    print(f"  Chance value for that (1 - lung area)      {mn:.3f}")
    print(f"  LIFT                                       {ml:+.3f} +- {ml_s:.3f}"
          f"   (paired t = {t_miss:+.2f}, |t|>2.78 = p<0.05)")

    print("\n" + "-" * 76)
    print("HEADROOM: what would taking away the field of view alone deliver?")
    print("-" * 76)
    hm, hm_s = cv_mean(per_fold, "peak_in_box_lungrestricted")
    bm, _ = cv_mean(per_fold, "peak_in_box")
    cm, cm_s = cv_mean(per_fold, "crop_headroom")
    nm, _ = cv_mean(per_fold, "headroom_null")
    vm, vm_s = cv_mean(per_fold, "headroom_vs_null")
    print(f"  Hits, maximum searched inside the lung only    {hm:.3f} +- {hm_s:.3f}")
    print(f"  compared with the current value                {bm:.3f}")
    print(f"  observed gain                                  {cm:+.3f} +- {cm_s:.3f}"
          f"   (t = {paired_t(per_fold, 'crop_headroom'):+.2f})")
    print(f"  gain of a RANDOM heat map                      {nm:+.3f}"
          "   <- plain area arithmetic")

    # Comparing gains invites the objection "ceiling effect": the model starts
    # higher and therefore has less room above it. Two lifts, each against its
    # own baseline, do not have that problem.
    lf, lf_s = cv_mean(per_fold, "lift_free")
    lr, lr_s = cv_mean(per_fold, "lift_restricted")
    print("\n  Stated free of ceiling effects, two lifts, each against its"
          " own null:")
    print(f"    free        {bm:.3f} - {cv_mean(per_fold, 'null_free')[0]:.3f}"
          f" = {lf:+.3f} +- {lf_s:.3f}")
    print(f"    restricted  {hm:.3f} - "
          f"{cv_mean(per_fold, 'null_restricted')[0]:.3f} = {lr:+.3f} +- {lr_s:.3f}")
    print(f"    DIFFERENCE  {vm:+.3f} +- {vm_s:.3f}   "
          f"(t = {paired_t(per_fold, 'lift_delta'):+.2f})")
    print("  Negative means: restricting to the lung makes the maximum LESS")
    print("  informative, not more. (Upper bound. A model trained on crops")
    print("  learns different weights.)")

    print("\n" + "=" * 76)
    print("LESART")
    print("=" * 76)
    if not mask_ok:
        # Order matters: a mask that is too small or displaced produces the
        # outside finding by itself. The interpretation would be circular.
        print("  NO INTERPRETATION as long as the mask controls are not passed.")
        print("  The outside share and the headroom are then artefacts of the")
        print("  segmentation, not a finding about the model. Repair the mask")
        print("  first (rsna_mask_sweep.py compares variants in seconds), then")
        print("  come back here.")
    elif np.isnan(ml):
        print("  No failures in the sample, so there is nothing to decide.")
    elif abs(t_miss) < 2.78:
        print(f"  The outside share ({mo:.3f}) is indistinguishable from its"
              f" chance value ({mn:.3f})")
        print(f"  and the lift is {ml:+.3f} (paired t = {t_miss:+.2f}).")
        print("  On failures the model does not look out of the lung in any")
        print("  systematic way. Its maximum is then simply uninformative.")
        print("  -> The mechanism the crop hope rests on is not there.")
        print("  -> Do not start step 3 on this basis.")
    elif ml > 0 and vm > 0.02:
        print("  The outside share exceeds its chance value AND the headroom")
        print("  exceeds the purely geometric gain. Together they are the")
        print("  mechanism a crop addresses: attention outside the lung.")
        print("  -> Step 3 (crop, 5 folds PAIRED per fold) is justified.")
    else:
        print("  Mixed: the outside share deviates from chance, but the headroom")
        print("  stays within plain area arithmetic. If measured at all, then")
        print("  paired per fold. A comparison of means could not separate an")
        print("  effect of this size from the spread across folds.")
    print("=" * 76)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, default=Path("data/rsna/png512"))
    p.add_argument("--masks", type=Path, default=Path("data/rsna/masks224"))
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--pred-dir", type=Path, default=Path("predictions_rsna"))
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--n", type=int, default=120,
                   help="images per fold (0 = all from the CAM CSV)")
    p.add_argument("--cache-heat", action="store_true",
                   help="cache heat maps as float16, which makes mask variants free")
    p.add_argument("--out", type=Path,
                   default=Path("predictions_rsna/cam_lung.csv"))
    args = p.parse_args(argv)

    if not Path(args.masks).exists():
        print(f"ERROR: mask directory missing: {args.masks}")
        print("  First:   python rsna_make_masks.py "
              "--ids-from \"predictions_rsna/cam_f*_s0.csv\" "
              "--raw-cache data/rsna/unet_raw256.npz")
        return 2

    frames, per_fold = [], []
    for f in args.folds:
        print(f"\nFold {f} (CPU, a few seconds per image)...")
        df = run_fold(f, args)
        if df.empty:
            continue
        frames.append(df)
        s = summarise(df)
        s["fold"] = f
        per_fold.append(s)
        print(f"  in box {s['peak_in_box']:.3f} | in lung {s['peak_in_lung']:.3f} "
              f"(chance {s['lung_area']:.3f}) | failure outside "
              f"{s['miss_outside_lung']:.3f} (chance {s['miss_outside_null']:.3f})")

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(args.out, index=False)
        pd.DataFrame(per_fold).to_csv(
            args.out.with_name(args.out.stem + "_byfold.csv"), index=False)
        print(f"\nRaw data: {args.out}")

    report(per_fold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
