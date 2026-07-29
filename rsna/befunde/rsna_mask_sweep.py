"""
Step 9c: was the headroom a mask artefact? Mask variants in seconds.

Why this script exists
----------------------
The diagnostic run in `rsna_cam_lung_check.py` left two results standing side
by side:

  * The outside finding does not carry. "80.7 % of the failures point out of
    the lung" stands against a chance value of 79.2 %, a lift of +0.014 at
    t = 0.32. Nothing.
  * Restricting the maximum of the Grad-CAM heat map, the map of the image
    regions that drove the model's score, to the lung nevertheless raises the
    hit rate by +0.080, and it does so in 5 of 5 folds.

The second result hangs on a mask that is measurably too small: lung area
0.210 instead of the anatomically expected 0.30-0.40, with 28.5 % of the
bounding-box area outside it. A mask that is too small produces both effects
by itself. It takes space away from a noise maximum, entirely without any
merit on the part of the model.

The cross-check is therefore to enlarge the mask step by step and watch what
happens to the headroom.

  * If the headroom shrinks towards the geometric null value as soon as the
    mask is anatomically plausible -> it was an artefact. The crop is settled,
    11.5 h are saved, and the result is a statement, not a failure.
  * If it persists -> the field of view really is the problem, and the crop
    run is justified.

Why a variant costs seconds
---------------------------
Two things are expensive: the U-Net forward pass, one sweep of the
segmentation network over every image (15 min), and the Grad-CAM computation
(20 min). Neither depends on the refinement. Both are cached once, the raw
U-Net output as packed bits (`rsna_make_masks.py --raw-cache`) and the heat
maps as float16 (`rsna_cam_lung_check.py --cache-heat`). Every further variant
after that is morphology plus a few argmax calls.

No masks are written to disk. A sweep that overwrote the mask directory as a
side effect would be a silent trap for the next run.

Interpreting the output
-----------------------
One table row per refinement variant (mode:dilation in pixels). All values are
means +- SD over the folds, the five patient splits the model was trained and
validated on; the DIFF column carries a paired t across folds (df = k-1, so
|t| > 2.78 is the 5 % bound with five folds). Read the rows as a series, not
one at a time. What matters is how the numbers move as the mask grows.

  * Lunge and boxInL are the mask controls. A variant that fails them (lung
    area < 0.26, box in lung < 0.60) cannot support any statement about the
    model; its outside share and its headroom follow from the geometry of the
    mask alone.
  * DIFF is the headroom stated free of ceiling effects: the restricted lift
    minus the free lift, each measured against its own null. Zero or negative
    means that restricting to the lung buys nothing beyond area arithmetic.

The decision. If, among the variants that pass both controls, DIFF stays at
the level of the purely geometric gain and the outside lift stays
indistinguishable from chance, then the headroom of the diagnostic run was a
mask artefact and the crop run is dropped: 11.5 h saved, with the finding
"segmentation contributes nothing here" as the reportable result. Only if both
survive a plausible mask does the crop have a mechanism to address, namely
attention that a narrower field of view would take away, and it then has to be
measured paired per fold.

CLI:
  # Prerequisite (once each):
  #   python rsna_make_masks.py --ids-from "predictions_rsna/cam_f*_s0.csv" \
  #       --raw-cache data/rsna/unet_raw256.npz
  #   python rsna_cam_lung_check.py --folds 0 1 2 3 4 --n 120 --cache-heat

  python rsna_mask_sweep.py
  python rsna_mask_sweep.py --variants none:0 default:0 default:6 hull:0 hull:6
"""

from __future__ import annotations

import _repo_path  # noqa: F401  (setzt sys.path fuer die Nachbarordner)

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_cam_lung_check import analyse_one, box_mask, cv_mean, paired_t, summarise
from rsna_make_masks import refine_variant, to_out, unpack_masks

DEFAULT_VARIANTS = ["none:0", "default:0", "default:4", "default:8",
                    "hull:0", "hull:4", "hull:8"]

BOX_SPACE = 1024        # bounding boxes live in the original DICOM grid


def load_boxes(csv_dir: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    """Deliberate copy of `rsna_train.load_boxes`.

    The sweep is post-processing on two caches. It runs no model and needs no
    torch. `rsna_train`, however, imports torch at module level, so importing
    from there would chain the sweep to a heavy and here unnecessary
    dependency, and make it unusable on any machine without a GPU stack.

    `test_rsna_masks.test_load_boxes_matches_train` guards the two versions
    against drifting apart: if torch is present, the result is compared
    against the original.
    """
    df = pd.read_csv(Path(csv_dir) / "stage_2_train_labels.csv")
    df = df[df["Target"] == 1].dropna(subset=["x", "y", "width", "height"])
    out: dict[str, list] = {}
    for pid, x, y, w, h in df[["patientId", "x", "y", "width", "height"]].values:
        out.setdefault(pid, []).append((float(x), float(y), float(w), float(h)))
    return out


def parse_variant(s: str) -> tuple[str, int]:
    """'hull:4' -> ('hull', 4). A function of its own, because a silent parse
    error here would make the whole table useless without anyone noticing."""
    if ":" not in s:
        return s, 0
    mode, _, px = s.partition(":")
    return mode, int(px)


def load_heat_cache(pred_dir: Path, fold: int, seed: int):
    p = Path(pred_dir) / f"cam_heat_f{fold}_s{seed}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return [str(s) for s in z["ids"]], z["heat"]


def run_variant(mode: str, dilate: int, raw_lookup: dict[str, np.ndarray],
                heat_by_fold: dict[int, tuple[list[str], np.ndarray]],
                boxes: dict, size: int, box_space: int) -> list[dict]:
    """Evaluate one refinement setting across all folds.

    Rebuilds the lung mask for that setting from the cached raw U-Net output,
    scores the cached heat maps against it and returns one summary row per
    fold. The per-fold rows are what makes the paired comparison possible. A
    pooled number would hide how much of the effect is fold difficulty.
    """
    per_fold = []
    for fold, (ids, heats) in sorted(heat_by_fold.items()):
        rows = []
        for pid, heat in zip(ids, heats):
            raw = raw_lookup.get(pid)
            if raw is None:
                continue
            lung = to_out(refine_variant(raw, mode, dilate), size) > 127
            r = analyse_one(np.asarray(heat, dtype=np.float32),
                            box_mask(boxes.get(pid, []), size, box_space), lung)
            if r is not None:
                rows.append(r)
        if not rows:
            continue
        s = summarise(pd.DataFrame(rows))
        s["fold"] = fold
        per_fold.append(s)
    return per_fold


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-cache", type=Path, default=Path("data/rsna/unet_raw256.npz"))
    p.add_argument("--pred-dir", type=Path, default=Path("predictions_rsna"))
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                   help="mode:dilation, e.g. hull:4")
    p.add_argument("--out", type=Path, default=Path("predictions_rsna/mask_sweep.csv"))
    args = p.parse_args(argv)

    if not args.raw_cache.exists():
        print(f"ERROR: raw cache missing: {args.raw_cache}")
        print('  python rsna_make_masks.py --ids-from "predictions_rsna/cam_f*_s0.csv" '
              f'--raw-cache {args.raw_cache}')
        return 2

    z = np.load(args.raw_cache, allow_pickle=False)
    raw_ids = [str(s) for s in z["ids"]]
    raw_lookup = {pid: unpack_masks(z["packed"][i:i + 1])[0]
                  for i, pid in enumerate(raw_ids)}
    print(f"Raw cache: {len(raw_lookup)} U-Net outputs")

    heat_by_fold = {}
    for f in args.folds:
        got = load_heat_cache(args.pred_dir, f, args.seed)
        if got is None:
            print(f"  Fold {f}: no heat cache, skipped.")
            continue
        heat_by_fold[f] = got
    if not heat_by_fold:
        print("ERROR: no heat cache found.")
        print("  python rsna_cam_lung_check.py --folds 0 1 2 3 4 --n 120 --cache-heat")
        return 2
    print(f"Heat cache: {sum(len(v[0]) for v in heat_by_fold.values())} maps "
          f"from {len(heat_by_fold)} folds\n")

    boxes = load_boxes(args.csv)

    rows = []
    for v in args.variants:
        mode, dil = parse_variant(v)
        pf = run_variant(mode, dil, raw_lookup, heat_by_fold, boxes,
                         args.size, BOX_SPACE)
        if not pf:
            continue
        rec = {"variante": v}
        for k in ("lung_area", "box_in_lung", "peak_in_box", "peak_in_lung",
                  "miss_outside_lung", "miss_outside_null", "miss_outside_lift",
                  "crop_headroom", "headroom_null", "headroom_vs_null",
                  "lift_free", "lift_restricted", "lift_delta"):
            m, s = cv_mean(pf, k)
            rec[k] = m
            rec[k + "_sd"] = s
        rec["t_miss"] = paired_t(pf, "miss_outside_lift")
        rec["t_headroom_vs_null"] = paired_t(pf, "headroom_vs_null")
        rows.append(rec)
        print(f"  {v:<12} lung area {rec['lung_area']:.3f}  "
              f"box_in_lung {rec['box_in_lung']:.3f}  "
              f"headroom {rec['crop_headroom']:+.3f}")

    if not rows:
        print("No variant could be evaluated.")
        return 1

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # ---- Table ------------------------------------------------------------
    print("\n" + "=" * 96)
    print("MASK VARIANTS: what changes once the mask becomes plausible?")
    print("=" * 96)
    print(f"{'Variante':<12}{'Lunge':>8}{'boxInL':>8}{'aussen':>9}{'Zufall':>9}"
          f"{'Vorspr':>9}{'VorFrei':>10}{'VorBeschr':>11}{'DIFF':>8}{'t':>8}")
    print("-" * 96)
    for _, r in df.iterrows():
        print(f"{r['variante']:<12}{r['lung_area']:>8.3f}{r['box_in_lung']:>8.3f}"
              f"{r['miss_outside_lung']:>9.3f}{r['miss_outside_null']:>9.3f}"
              f"{r['miss_outside_lift']:>+9.3f}{r['lift_free']:>+10.3f}"
              f"{r['lift_restricted']:>+11.3f}{r['lift_delta']:>+8.3f}"
              f"{r['t_headroom_vs_null']:>8.2f}")
    print("-" * 96)
    print("  Lunge     = mask area (anatomically ~0.30-0.40 to be expected)")
    print("  boxInL    = share of the bounding box inside the mask (control against")
    print("              undersegmentation; low = the mask cuts the pathology away)")
    print("  aussen    = failures with max outside the lung, chance = 1 - lung area")
    print("  VorFrei   = hit rate MINUS box area")
    print("  VorBeschr = hit rate(lung only) MINUS (box AND lung)/lung")
    print("  DIFF      = VorBeschr - VorFrei, paired per fold. Both are distances")
    print("              to their OWN null, hence comparable free of ceiling effects.")
    print("              Negative = restricting to the lung makes the maximum")
    print("              less informative, not more.")

    # ---- Interpretation ---------------------------------------------------
    ok = df[(df.lung_area >= 0.26) & (df.box_in_lung >= 0.60)]
    print("\n" + "=" * 96)
    print("LESART")
    print("=" * 96)
    if ok.empty:
        print("  NO variant passes both mask controls (lung area >= 0.26 AND")
        print("  box_in_lung >= 0.60). The U-Net undersegments on RSNA more strongly")
        print("  than hull and dilation can repair.")
        print("  -> The next question is then not the crop, but whether the")
        print("     segmentation is viable in this project at all.")
        print("     That is a reportable result, not a failure.")
    else:
        best = ok.loc[ok.headroom_vs_null.idxmax()]
        print(f"  Best variant that passes both controls: {best['variante']}")
        print(f"    lung area {best['lung_area']:.3f} | "
              f"box_in_lung {best['box_in_lung']:.3f}")
        print(f"    lift free {best['lift_free']:+.3f} against restricted "
              f"{best['lift_restricted']:+.3f}  ->  difference "
              f"{best['lift_delta']:+.3f} (t = {best['t_headroom_vs_null']:+.2f})")
        print(f"    outside lift {best['miss_outside_lift']:+.3f} "
              f"(t = {best['t_miss']:+.2f})")
        if best["headroom_vs_null"] > 0.02 and abs(best["t_miss"]) > 2.78:
            print("  -> Both carry: the crop run is justified. Measure it PAIRED per")
            print("     fold, otherwise a difference of 0.005 means nothing.")
        else:
            print("  -> Even with a plausible mask there is no surplus over plain")
            print("     area arithmetic. The headroom was geometry, not field of")
            print("     view. Step 3 is dropped, 11.5 h saved, and the finding")
            print("     'segmentation contributes nothing here' belongs on record.")
    print("=" * 96)
    print(f"\nTable: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
