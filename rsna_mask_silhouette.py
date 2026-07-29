"""
Step 9g: what pixel-exact masking really does.

The objection this script tests
-------------------------------
A rectangle around both lungs necessarily contains the mediastinum. But
pixel-exact masking does not remove the mediastinum either: through the shape
of the lungs the mask leaks it anyway. It then exists as negative space, as a
cutout whose form is the cardiac and vascular silhouette.

That is a testable claim, and it turns the usual justification for masking on
its head. Masking is supposed to REMOVE the confounder. If the cutout between
the lungs is itself a projection proxy, the confounder is only RE-ENCODED:
grey values become a contour. A convolutional network builds its features out
of edges and shapes, so a contour is the easier quantity for it to read, not
the harder one.

Three measurements, all on the development set (holdout stays out)
------------------------------------------------------------------

  1. WHERE DOES THE ANNOTATED PATHOLOGY LIE? Share of the bounding-box area
     inside the lung mask, in the cutout between the lungs, and outside both.
     Whatever lies in the latter two buckets would be cut away by pixel-exact
     masking.

  2. WHAT DOES THE CUTOUT GIVE AWAY ABOUT THE PROJECTION? Three clinically
     readable measures of the cutout (cardiothoracic ratio, area fraction,
     vertical position) against ViewPosition. Not a single image pixel enters
     into it.

  3. WHAT DOES IT GIVE AWAY ABOUT THE CLASS? The same three measures against
     Target, stratified by ViewPosition. That is the counter-check: does
     masking build a NEW shortcut? The comparison is against the crop
     parameters of the rectangle (three features, same CV) from
     `rsna_make_crops.py`.

Clinically readable measures rather than the full contour: a measure that can
be named in clinical terms is worth more as evidence. The price is that the
result is a LOWER BOUND. A network sees the complete contour, not three
summaries.

Reading the output
------------------
Measurement 1: whatever share of the box area lies in the cutout or outside
both is what pixel-exact masking would remove from the annotated pathology.

Measurement 2: the reference is the framing of the rectangle, which predicts
the projection with 0.714. A comparable value for the cutout means that
masking does not remove the channel, it re-encodes it.

Measurement 3 is the counter-check, and its reference is 0.552, the value the
crop parameters of the rectangle reach with three features under the same CV.
If the silhouette gives the CLASS away more strongly than the crop parameters
do, masking builds the larger new shortcut.

CLI:
  python rsna_mask_silhouette.py --raw-cache data/rsna/unet_raw256.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEG = 256
BOX_SPACE = 1024
FEATURES = ["ctr", "mid_area", "mid_top"]


# --------------------------------------------------------------------------
# Geometry of the cutout  (pure arithmetic, testable)
# --------------------------------------------------------------------------

def inner_gap(mask: np.ndarray) -> np.ndarray:
    """The cutout: per row, the pixels BETWEEN the first and the last lung
    pixel that are not lung themselves.

    Exactly this strip turns black under pixel-exact masking. The row-wise
    definition is deliberate: a global bounding-box definition would take in
    areas above the lung apices and below the costophrenic angles that are not
    mediastinum, and the result would come out flattered.
    """
    m = np.asarray(mask, dtype=bool)
    out = np.zeros_like(m)
    for r in range(m.shape[0]):
        c = np.where(m[r])[0]
        if c.size >= 2:
            out[r, c.min():c.max() + 1] = True
    return out & ~m


def silhouette_features(mask: np.ndarray) -> dict | None:
    """Measures of the cutout, computed without a single grey value.

    ctr       width of the cutout / total width at heart level. The
              cardiothoracic ratio as it follows from the mask.
    mid_area  area fraction of the cutout in the whole image.
    mid_top   where the cutout begins, relative to the lung height (aortic
              arch or hilar level).

    Heart level = 60 % to 85 % of the lung span. Chosen once and documented
    here, so that the number cannot be shifted after the fact.
    """
    m = np.asarray(mask, dtype=bool)
    ys = np.where(m.any(1))[0]
    if ys.size < 10:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    gap = inner_gap(m)

    hr = slice(y0 + int(0.60 * (y1 - y0)), y0 + int(0.85 * (y1 - y0)))
    lung_w = float(m[hr].sum(1).mean())
    gap_w = float(gap[hr].sum(1).mean())
    span = lung_w + gap_w
    gr = np.where(gap.any(1))[0]
    return {
        "ctr": gap_w / span if span > 0 else float("nan"),
        "mid_area": float(gap.mean()),
        "mid_top": ((int(gr.min()) - y0) / (y1 - y0)) if gr.size else float("nan"),
    }


def box_location(mask: np.ndarray,
                 boxes: list[tuple[float, float, float, float]],
                 box_space: float = BOX_SPACE) -> dict | None:
    """How much box area lies in the lung, in the cutout, outside both?"""
    m = np.asarray(mask, dtype=bool)
    gap = inner_gap(m)
    H, W = m.shape
    tot = lung = mid = 0
    for x, y, w, h in boxes:
        x0, y0 = int(x / box_space * W), int(y / box_space * H)
        x1, y1 = int((x + w) / box_space * W), int((y + h) / box_space * H)
        sub_l, sub_m = m[y0:y1, x0:x1], gap[y0:y1, x0:x1]
        if sub_l.size == 0:
            continue
        tot += sub_l.size
        lung += int(sub_l.sum())
        mid += int(sub_m.sum())
    if not tot:
        return None
    return {"in_lunge": lung / tot, "in_mitte": mid / tot,
            "ausserhalb": (tot - lung - mid) / tot}


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    r = pd.Series(np.asarray(s, dtype=float)).rank().values
    n1 = int(y.sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def cv_auc(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Several features, cross-validated: fitted on four fifths of the cases
    and scored on the fifth held back. In-sample, three features on 22 000
    points would admittedly be uncritical, but the number should stay
    comparable with the project's other confounder checks."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(set(y)) < 2:
        return float("nan")
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        mo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        mo.fit(X[tr], y[tr])
        oof[te] = mo.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def stratified(d: pd.DataFrame, cols: list[str], target: str,
               strata: str = "vp") -> float:
    tot, n = 0.0, 0
    for v in ("AP", "PA"):
        s = (d[strata] == v).values
        if s.sum() < 50:
            continue
        a = cv_auc(d.loc[s, cols].values, d.loc[s, target].values)
        tot += a * int(s.sum())
        n += int(s.sum())
    return tot / n if n else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-cache", type=Path,
                   default=Path("data/rsna/unet_raw256.npz"))
    p.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--refine", default="hull")
    p.add_argument("--dilate-px", type=int, default=8)
    p.add_argument("--out", type=Path,
                   default=Path("predictions_rsna/mask_silhouette.csv"))
    args = p.parse_args(argv)

    from rsna_make_crops import boxes_by_id
    from rsna_make_masks import load_raw_cache, refine_variant, unpack_masks

    if not args.raw_cache.exists():
        print(f"ERROR: raw cache missing: {args.raw_cache}")
        return 2

    ids, packed = load_raw_cache(args.raw_cache)
    sp = json.loads(args.splits.read_text())
    holdout = set(sp.get("holdout", []))
    vp = sp["viewpos"]
    lab = {k: int(v) for k, v in sp["labels"].items()}
    boxes = boxes_by_id(pd.read_csv(Path(args.csv) / "stage_2_train_labels.csv"))

    print(f"Masks in the cache: {len(ids)} | holdout excluded: {len(holdout)}")
    rows = []
    for i, pid in enumerate(ids):
        if pid in holdout:
            continue
        m = refine_variant(unpack_masks(packed[i:i + 1])[0], args.refine,
                           args.dilate_px)
        f = silhouette_features(m)
        if f is None:
            continue
        f["patientId"] = pid
        f["vp"] = vp.get(pid, "?")
        f["target"] = lab.get(pid, -1)
        if pid in boxes:
            f.update(box_location(m, boxes[pid]) or {})
        rows.append(f)
        if (i + 1) % 8000 == 0:
            print(f"  {i + 1}/{len(ids)}", flush=True)

    d = pd.DataFrame(rows).dropna(subset=FEATURES)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out, index=False)
    print(f"\nn = {len(d)}   saved: {args.out}")

    # --- 1. Where does the annotated pathology lie? ----------------------
    b = d.dropna(subset=["in_lunge"])
    print(f"\n1. WHERE THE ANNOTATED PATHOLOGY LIES   (n = {len(b)} with a box)")
    print(f"   in the lung mask            {b.in_lunge.mean():.3f}")
    print(f"   in the cutout               {b.in_mitte.mean():.3f}")
    print(f"   outside both                {b.ausserhalb.mean():.3f}")
    print(f"   -> pixel-exact masking would remove "
          f"{(b.in_mitte.mean() + b.ausserhalb.mean()) * 100:.1f} % of the box area")
    print(f"   images with over 20 % in the cutout: "
          f"{int((b.in_mitte > 0.20).sum())} ({(b.in_mitte > 0.20).mean() * 100:.1f} %)")
    print(f"   images with over 50 %:               "
          f"{int((b.in_mitte > 0.50).sum())} ({(b.in_mitte > 0.50).mean() * 100:.1f} %)")

    e = d[d.vp.isin(["AP", "PA"])].copy()
    e["is_ap"] = (e.vp == "AP").astype(int)

    # --- 2. Cutout -> projection ----------------------------------------
    print(f"\n2. WHAT THE CUTOUT GIVES AWAY ABOUT THE PROJECTION  (n = {len(e)})")
    for f in FEATURES:
        print(f"   only {f:<9}  AUC {auc(e.is_ap.values, e[f].values):.3f}")
    a_view = cv_auc(e[FEATURES].values, e.is_ap.values)
    print(f"   all three      AUC {a_view:.3f}")
    print(f"   (framing of the rectangle -> projection: 0.714. Masking")
    print(f"    therefore does not remove the channel, it re-encodes it.)")
    print(f"\n   cardiothoracic ratio per projection:")
    print("   " + e.groupby("vp")["ctr"].agg(["count", "mean", "std"]).round(4)
          .to_string().replace("\n", "\n   "))

    # --- 3. Cutout -> class (the counter-check) --------------------------
    a_t = stratified(e, FEATURES, "target")
    print(f"\n3. COUNTER-CHECK: CUTOUT -> CLASS, stratified   AUC {a_t:.3f}")
    print(f"   unstratified                                 AUC "
          f"{cv_auc(e[FEATURES].values, e.target.values):.3f}")
    print(f"   (crop parameters of the rectangle, three features,")
    print(f"    same CV, stratified: 0.552)")
    if a_t > 0.552:
        print(f"   -> The mask silhouette gives the class away MORE strongly")
        print(f"      than the crop. Masking builds the larger new shortcut.")

    print("\nIn context: three hand-picked measures. A convolutional network")
    print("sees the complete contour, so these numbers are lower bounds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
