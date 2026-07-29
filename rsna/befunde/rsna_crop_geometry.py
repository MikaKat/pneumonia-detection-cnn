"""
Step 9d: does a rectangular crop attack the ViewPosition confounder?

The question
------------
The proposal under test: crop to the BOUNDING RECTANGLE plus margin rather
than masking pixel-exactly to the lung mask. A rectangle tolerates segmentation
error better, because its edges are set by the lung apices and the costophrenic
angles, and the U-Net finds those reliably even when it loses a consolidation
in the middle.

The obvious mechanism would be effective resolution, and that one is already
bounded: the crop would bring 1.1x to 1.4x linear zoom, and the model loses
only 0.016 AUC at 0.45x resolution. Expected effect therefore about 0.005,
below the fold spread of 0.015. AUC is the probability that a random positive
case is scored above a random negative one, with 0.5 as chance.

The second mechanism is stronger, and this script tests it: the crop normalises
the framing. AP supine images are framed differently from PA upright images,
with more shoulder, more abdomen, a different distance and different centring.
If that holds, the framing carries information about the projection, and at
+0.044 the projection is the largest known nuisance variable (ViewPosition ->
Target: AUC 0.706; the model score predicts the projection with AUC 0.808, so
it reads the projection off the image).

A crop that normalises the framing away would close this channel. That would be
a far better reason than resolution.

Two measurements, both required
-------------------------------
1. AUC(rectangle geometry -> ViewPosition). High means the framing is a
   projection proxy and the crop has something to normalise away. Near 0.5
   means there is nothing to remove, and only the weak resolution argument
   remains.

2. AUC(rectangle geometry -> Target). The counter-check. If the crop parameters
   themselves give away the class, cropping along those parameters builds in a
   NEW shortcut, which is the mistake that happened on Kermany (mask area AUC
   0.255). This number must lie near 0.5, otherwise the crop is harmful no
   matter what measurement 1 says.

One design consequence is already settled
-----------------------------------------
Scaling a non-square rectangle to 224x224 distorts the image by an amount that
varies with the aspect ratio. The aspect ratio then survives the crop as a
distortion signature, so the channel would not be closed, merely re-encoded.
That is why `aspect` is reported separately here: if it alone is already a good
projection proxy, the crop has to be extended to a SQUARE rectangle (grow the
shorter side).

Interpreting the output
-----------------------
Measurement 1 states what a crop could remove. A high AUC means the framing is
a projection proxy and the crop has something to normalise away; an AUC near
0.5, the null value, means there is nothing to remove and the case for the crop
falls back on the weak resolution argument alone. Measurement 2 runs the other
way and is read stratified by projection, so that the known confounder cannot
simply pass through: if the crop parameters predict the CLASS within a single
projection, the crop would build a new shortcut, and that argues against it
regardless of measurement 1. So the crop is warranted only if measurement 1 is
high AND stratified measurement 2 stays near 0.5. The `aspect` row is read on
its own: a good projection proxy there means the rectangle must be squared
before resizing, or the channel is only re-encoded as distortion.

CLI:
  python rsna_crop_geometry.py                       # uses the raw cache
  python rsna_crop_geometry.py --refine hull --dilate-px 8 --pad 0.05
"""

from __future__ import annotations

import _repo_path  # noqa: F401  (setzt sys.path fuer die Nachbarordner)

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_make_masks import refine_variant, unpack_masks

FEATURES = ["y0", "y1", "x0", "x1", "height", "width", "aspect", "cy", "cx",
            "area"]


def rect_features(mask: np.ndarray, pad: float) -> dict | None:
    """The parameters a crop would be taken by, as a feature vector.

    Returns the padded bounding rectangle of the lung mask, described by ten
    numbers normalised to image size (edges, height, width, aspect ratio,
    centre, area), or None for an empty mask.

    Exactly these quantities determine the crop. If a classifier can predict
    the projection from them, or worse the class, then the crop itself carries
    that information, which is what the two measurements downstream quantify.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    H, W = mask.shape
    ph = pad * (ys.max() - ys.min() + 1)
    pw = pad * (xs.max() - xs.min() + 1)
    y0, y1 = max(ys.min() - ph, 0), min(ys.max() + ph, H - 1)
    x0, x1 = max(xs.min() - pw, 0), min(xs.max() + pw, W - 1)
    h, w = y1 - y0 + 1, x1 - x0 + 1
    return {"y0": y0 / H, "y1": y1 / H, "x0": x0 / W, "x1": x1 / W,
            "height": h / H, "width": w / W, "aspect": w / h,
            "cy": (y0 + y1) / 2 / H, "cx": (x0 + x1) / 2 / W,
            "area": (h * w) / (H * W)}


def cv_auc(X: np.ndarray, y: np.ndarray, seed: int = 0, folds: int = 5) -> float:
    """AUC of a logistic model under group-free cross-validation.

    Cross-validation fits on part of the data and scores the part held out,
    rather than fitting and scoring on the same points: ten features on 1500
    points would give a flattering in-sample number. Every RSNA patientId occurs
    exactly once, so a stratified split suffices here. There are no groups that
    could be torn apart, hence no leak across folds.

    The returned out-of-fold AUC has 0.5 as its null value: 0.5 means the
    features carry no information about the label, and the further above 0.5,
    the more of the label the crop parameters alone already determine.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(set(y)) < 2:
        return float("nan")
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def stratified_auc(df: pd.DataFrame, target: np.ndarray,
                   strata: np.ndarray) -> tuple[float, dict]:
    """AUC within each projection, then averaged weighted by stratum size.

    Stratification is mandatory here because `ViewPosition -> Target` has AUC
    0.706 and `geometry -> ViewPosition` 0.745. The geometry can therefore
    predict the class entirely THROUGH the projection, without a single
    contribution of its own. An unstratified number cannot distinguish "new
    shortcut" from "known confounder passing through", and only the first case
    would argue against the crop. Unstratified the value reaches 0.621, which
    reads as a warning; stratified, 0.541 of that remains.

    Returns the weighted mean plus the per-projection AUCs. Read against 0.5:
    a stratified value near 0.5 means the geometry carries no signal of its
    own, a clearly higher one means the crop would introduce a new shortcut.
    """
    per, tot, n = {}, 0.0, 0
    for s in sorted(set(strata)):
        sel = strata == s
        if len(set(target[sel])) < 2 or sel.sum() < 50:
            continue
        a = cv_auc(df[FEATURES].values[sel], target[sel])
        per[s] = (a, int(sel.sum()))
        tot += a * sel.sum()
        n += int(sel.sum())
    return (tot / n if n else float("nan")), per


def report_block(name: str, df: pd.DataFrame, target: np.ndarray,
                 refs: list[tuple[str, float]]) -> float:
    print(f"\n  {name}   (n = {len(df)}, positive = {int(target.sum())})")
    full = cv_auc(df[FEATURES].values, target)
    print(f"    all 10 geometry features          AUC {full:.3f}")
    for f in ("aspect", "area", "height", "cy"):
        a = cv_auc(df[[f]].values, target)
        print(f"    only {f:<28} AUC {a:.3f}")
    for label, val in refs:
        print(f"    (reference: {label} {val:.3f})")
    return full


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-cache", type=Path, default=Path("data/rsna/unet_raw256.npz"))
    p.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    p.add_argument("--refine", default="hull")
    p.add_argument("--dilate-px", type=int, default=8)
    p.add_argument("--pad", type=float, default=0.05)
    p.add_argument("--out", type=Path,
                   default=Path("predictions_rsna/crop_geometry.csv"))
    args = p.parse_args(argv)

    if not args.raw_cache.exists():
        print(f"ERROR: raw cache missing: {args.raw_cache}")
        return 2

    sp = json.loads(args.splits.read_text())
    labels = {k: int(v) for k, v in sp["labels"].items()}
    vpmap = sp["viewpos"]

    z = np.load(args.raw_cache, allow_pickle=False)
    ids = [str(s) for s in z["ids"]]
    packed = z["packed"]
    print(f"raw cache: {len(ids)} masks | refine={args.refine} "
          f"dilate={args.dilate_px} pad={args.pad}")

    rows = []
    for i, pid in enumerate(ids):
        m = refine_variant(unpack_masks(packed[i:i + 1])[0], args.refine,
                           args.dilate_px)
        f = rect_features(m, args.pad)
        if f is None:
            continue
        f["patientId"] = pid
        f["target"] = labels.get(pid, -1)
        f["vp"] = vpmap.get(pid, "?")
        rows.append(f)
    d = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out, index=False)

    print(f"\ncrop rectangle: area {d.area.mean():.3f} +- {d.area.std():.3f}"
          f" | linear zoom {(1 / np.sqrt(d.area)).mean():.2f}x")
    print(f"aspect ratio (width/height): {d.aspect.mean():.3f} "
          f"+- {d.aspect.std():.3f}")

    print("\n" + "=" * 70)
    print("MEASUREMENT 1: does the framing carry information about the projection?")
    print("=" * 70)
    dv = d[d.vp.isin(["AP", "PA"])]
    vp_auc = report_block("geometry -> ViewPosition (AP=1)", dv,
                          (dv.vp == "AP").astype(int).values,
                          [("model score -> ViewPosition", 0.808)])

    print("\n" + "=" * 70)
    print("MEASUREMENT 2: does the crop itself give away the class?")
    print("=" * 70)
    dt = d[d.target >= 0]
    if dt.target.nunique() < 2:
        t_auc = float("nan")
        print(f"\n  NOT DETERMINABLE: the cache holds only target="
              f"{int(dt.target.iloc[0])}.")
        print("  Grad-CAM was measured on positive images only, so there are no")
        print("  negatives in the raw cache. Masks are missing for this number:")
        print("    python rsna_make_masks.py --ids-from <CSV with negatives> \\")
        print("        --raw-cache data/rsna/unet_raw256.npz")
    else:
        report_block("geometry -> Target, UNSTRATIFIED", dt, dt.target.values,
                     [("ViewPosition alone -> Target", 0.706),
                      ("header baseline", 0.557)])

        # The unstratified number cannot be interpreted (see stratified_auc).
        # Only what is left within a single projection is decisive.
        ds = dt[dt.vp.isin(["AP", "PA"])]
        t_auc, per = stratified_auc(ds, ds.target.values, ds.vp.values)
        print("\n  Stratified by ViewPosition: does the geometry carry OWN signal?")
        for s, (a, n) in per.items():
            print(f"    only {s:<28} AUC {a:.3f}   (n = {n})")
        print(f"    weighted mean                     AUC {t_auc:.3f}"
              "   <- the decisive number")

    print("\n" + "=" * 70)
    print("LESART")
    print("=" * 70)
    asp_vp = cv_auc(dv[["aspect"]].values, (dv.vp == "AP").astype(int).values)
    if vp_auc >= 0.70:
        print(f"  The framing predicts the projection with AUC {vp_auc:.3f}, so")
        print("  there is something to normalise away. Two caveats:")
        print(f"  1. The aspect ratio alone already accounts for {asp_vp:.3f} of it,")
        print("     and that is anatomy: it stays visible in the pixels no matter")
        print(f"     how it is cut. Only the rest (~{vp_auc - asp_vp:+.3f}) is removable.")
        print("  2. The reported number is STRATIFIED anyway (0.845). The crop")
        print("     would barely move it. What it could change is the model's")
        print("     DEPENDENCE on the projection, measurable by whether")
        print("     'model score -> ViewPosition' drops from 0.808. That is a")
        print("     legitimate goal, but a different claim than 'better AUC'.")
    elif vp_auc >= 0.60:
        print(f"  Moderate (AUC {vp_auc:.3f}): the framing carries something, but little.")
        print("  The crop would close part of the channel. Borderline case.")
    else:
        print(f"  The framing barely predicts the projection (AUC {vp_auc:.3f}).")
        print("  There is nothing to normalise away. The crop then acts only via")
        print("  resolution, and that is bounded at ~0.005.")
        print("  -> Not a sufficient reason for 2.3 h per fold.")

    if np.isnan(t_auc):
        pass
    elif abs(t_auc - 0.5) > 0.08:
        print(f"\n  WARNING: the crop parameters give away the class even")
        print(f"  WITHIN a single projection (AUC {t_auc:.3f}). That is a new,")
        print("  independent shortcut, the same mistake as the mask area on")
        print("  Kermany (AUC 0.255). The crop is not usable in this form.")
    else:
        print(f"\n  Shortcut all clear: stratified, only {t_auc:.3f} remains.")
        print("  The unstratified value arises almost entirely because the")
        print("  geometry predicts the PROJECTION and the projection predicts")
        print("  the class. No new channel, just the known one passing through.")

    asp = asp_vp
    if asp >= 0.60:
        print(f"\n  DESIGN CONSEQUENCE: the aspect ratio alone predicts the")
        print(f"  projection with AUC {asp:.3f}. A non-square crop, stretched to")
        print("  224x224, re-encodes this channel as distortion instead of")
        print("  closing it. So extend to a SQUARE rectangle (grow the shorter")
        print("  side), do not scale the two sides independently.")
    print("=" * 70)
    print(f"\nfeatures: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
