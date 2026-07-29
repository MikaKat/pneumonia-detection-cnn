"""
Step 9f: does the crop settle the question it was built for?

The endpoints are fixed BEFORE the first crop run, so that the number which
looks best afterwards cannot become the one that gets reported. They are
hard-wired here and not chosen at evaluation time:

  PRIMARY   AUC(model score -> ViewPosition). AUC is the probability that a
            randomly drawn AP film scores above a randomly drawn PA one; at
            0.5 the score would say nothing about the projection.
            Baseline, measured on the existing predictions of the five
            folds: 0.8166 +- 0.0098. (The 0.808 in READMEforMe is fold 0
            alone; a single-fold number is of no use here.)
            If the value FALLS, the crop has lowered the dependence on the
            confounder. That is the claim being tested.

  SECONDARY stratified AUC, paired within the same fold. Baseline
            0.845 +- 0.015. It must not fall meaningfully. A crop that
            lowers the confounder and takes the discriminative power with
            it has gained nothing.

  CONTROL   Grad-CAM hit rate against the TRANSFORMED boxes. It is in
            results_rsna.csv as cam_hit / cam_hit_lift and is only carried
            along here.

PAIRED, SAME BUDGET. A fold is compared against the same fold, both with 8
epochs. A longer crop run against a short baseline measures epochs, not crop.

Why the direction of the primary endpoint counts and not its level:
ViewPosition -> Target has AUC 0.706, and the model score predicts the
projection with 0.8166. The model reads the projection off the image and can
guess part of the label through it without doing any radiology. The crop is
meant to close that path. If the number stays where it is, the path is still
open, regardless of what the AUC does.

How much decline is possible at all, recorded in advance
--------------------------------------------------------
The rectangle encloses both lungs, so the mediastinum stays in. The crop
therefore does not remove the AP/PA cues INSIDE THE THORAX: cardiac
enlargement in the supine position, the vascular pedicle and projected
scapulae survive unchanged. It can only close the EXTERNAL channel, where
the thorax sits in the image and how large it appears. Of that, the framing
carries 0.714 (`rsna_crop_geometry.py`).

From this follows the expectation, and it stands here BEFORE a single crop
number exists:

  * A PARTIAL decline is the methodologically expected result, not half a
    success.
  * A decline down to near 0.5 would be suspicious rather than welcome. It
    would mean that the projection had previously hung almost entirely on the
    framing and not on the anatomy. That is hard to believe and would first of
    all be a reason to check the crop itself.
  * No decline refutes the rationale of the crop. That too is a result, and
    one backed by evidence.

That the mediastinum stays in is not a concession but the better route:
`rsna_mask_silhouette.py` shows that pixel-exact masking does not remove the
projection channel (0.692 survive in the silhouette), that it builds a
STRONGER new shortcut in doing so (0.593 against 0.552) and that it destroys
12.9 % of the annotated pathology.

CLI:
  python rsna_crop_compare.py \
      --a predictions_rsna --b predictions_rsna_crop \
      --name-a Basislinie --name-b Zuschnitt
"""

from __future__ import annotations

import _repo_path  # noqa: F401  (setzt sys.path fuer die Nachbarordner)

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_cam_lung_check import cv_mean, paired_t

PRIMAER_BASIS = 0.8166          # from the existing predictions, 5 folds
SEKUNDAER_BASIS = 0.845


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC via rank sums, deliberately without sklearn. The averaged rank ties
    from `pd.Series.rank` are the only subtlety here."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    r = pd.Series(np.asarray(s, dtype=float)).rank().values
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def score_to_view(d: pd.DataFrame, col: str = "p_clean") -> float:
    """THE PRIMARY ENDPOINT: how well does the model score give away the
    projection?"""
    d = d[d["viewpos"].isin(["AP", "PA"])]
    if d.empty:
        return float("nan")
    return auc((d["viewpos"] == "AP").values, d[col].values)


def stratified_auc(d: pd.DataFrame, col: str = "p_clean",
                   min_n: int = 50) -> float:
    """AUC within each projection, averaged with weights by size.

    The same definition as in `rsna_train.stratified_scores`. It is recomputed
    here instead of read from results_rsna.csv, so that both variants are
    guaranteed to be scored with the same formula.
    """
    tot, n = 0.0, 0
    for v in ("AP", "PA"):
        sel = d["viewpos"] == v
        if sel.sum() < min_n or d.loc[sel, "y"].nunique() < 2:
            continue
        a = auc(d.loc[sel, "y"].values, d.loc[sel, col].values)
        tot += a * int(sel.sum())
        n += int(sel.sum())
    return tot / n if n else float("nan")


def read_fold(pred_dir: Path, fold: int, seed: int) -> pd.DataFrame | None:
    p = Path(pred_dir) / f"rsna_f{fold}_s{seed}.csv"
    return pd.read_csv(p) if p.exists() else None


def compare(dir_a: Path, dir_b: Path, folds: list[int], seed: int,
            col: str = "p_clean") -> list[dict]:
    """Evaluate both variants per fold. If one side is missing, the fold drops
    out entirely; a half-paired comparison is not a paired comparison."""
    rows = []
    for f in folds:
        a, b = read_fold(dir_a, f, seed), read_fold(dir_b, f, seed)
        if a is None or b is None:
            print(f"  Fold {f}: skipped "
                  f"({'A' if a is None else 'B'} missing)")
            continue
        if col not in a.columns or col not in b.columns:
            print(f"  Fold {f}: column {col} missing, skipped")
            continue
        if len(a) != len(b):
            print(f"  Fold {f}: WARNING, {len(a)} against {len(b)} images. "
                  f"The crop should deliver the same set.")
        r = {
            "fold": f,
            "view_a": score_to_view(a, col), "view_b": score_to_view(b, col),
            "strat_a": stratified_auc(a, col), "strat_b": stratified_auc(b, col),
            "auc_a": auc(a["y"].values, a[col].values),
            "auc_b": auc(b["y"].values, b[col].values),
        }
        r["view_d"] = r["view_b"] - r["view_a"]
        r["strat_d"] = r["strat_b"] - r["strat_a"]
        r["auc_d"] = r["auc_b"] - r["auc_a"]
        rows.append(r)
    return rows


def report(rows: list[dict], name_a: str, name_b: str) -> None:
    if not rows:
        print("\nNo fold complete, nothing to compare.")
        return

    print(f"\n{'Fold':>5}  {'Score->View':>22}  {'stratified AUC':>22}")
    print(f"{'':>5}  {name_a[:9]:>9} {name_b[:9]:>9} {'d':>6}  "
          f"{name_a[:9]:>9} {name_b[:9]:>9} {'d':>6}")
    for r in rows:
        print(f"{r['fold']:>5}  {r['view_a']:9.4f} {r['view_b']:9.4f} "
              f"{r['view_d']:+6.3f}  {r['strat_a']:9.4f} {r['strat_b']:9.4f} "
              f"{r['strat_d']:+6.3f}")

    print("\n--- PRIMARY: AUC(model score -> ViewPosition), should FALL ---")
    ma, sa = cv_mean(rows, "view_a")
    mb, sb = cv_mean(rows, "view_b")
    md, sd = cv_mean(rows, "view_d")
    t = paired_t(rows, "view_d")
    print(f"  {name_a:<12} {ma:.4f} +- {sa:.4f}   (reference {PRIMAER_BASIS:.4f})")
    print(f"  {name_b:<12} {mb:.4f} +- {sb:.4f}")
    print(f"  difference   {md:+.4f} +- {sd:.4f}   paired t = {t:+.2f}")
    if len(rows) < 2:
        print(f"  ONE FOLD ONLY. The t value is undefined, not zero, and the "
              f"+- 0.0000 above")
        print(f"  is an artefact of n = 1, not agreement. At least three folds "
              f"are needed")
        print(f"  before the difference of {md:+.4f} counts as more than a "
              f"direction.")
    else:
        print(f"  (with five folds, |t| > 2.78 is the 5 % limit)")

    print("\n--- SECONDARY: stratified AUC, must not collapse ---")
    ma, sa = cv_mean(rows, "strat_a")
    mb, sb = cv_mean(rows, "strat_b")
    md, sd = cv_mean(rows, "strat_d")
    t2 = paired_t(rows, "strat_d")
    print(f"  {name_a:<12} {ma:.4f} +- {sa:.4f}   (reference {SEKUNDAER_BASIS:.3f})")
    print(f"  {name_b:<12} {mb:.4f} +- {sb:.4f}")
    print(f"  difference   {md:+.4f} +- {sd:.4f}   paired t = {t2:+.2f}")

    print("\n--- Reading, fixed in advance ---")
    # The wording used to name the crop, because the crop was the only variant
    # this script had ever been pointed at. It then fired verbatim on the
    # --balance-view run and read as approval of an experiment whose
    # pre-registration says something else. The verdict now speaks about
    # "variant B", so the reader has to go and look up what B was rather than
    # being handed a conclusion written for a different question.
    view_d = cv_mean(rows, "view_d")[0]
    strat_d = cv_mean(rows, "strat_d")[0]
    # NaN compares false against everything, so an undefined secondary
    # endpoint would silently fall through to the "bought, not won" branch and
    # read like a measured statement. Say so instead.
    if view_d != view_d or strat_d != strat_d:
        print("  NO VERDICT: one of the two endpoints is undefined (a stratum")
        print("  with only one class, or no fold in common). Check the inputs.")
        return
    if view_d < -0.02 and strat_d > -0.015:
        print(f"  {name_b} lowers the dependence on the confounder without")
        print("  costing discriminative power on the SECONDARY endpoint.")
    elif view_d < -0.02:
        print(f"  {name_b} lowers the confounder, and the stratified AUC falls")
        print("  with it. Bought, not won. Report both numbers together.")
    elif abs(view_d) <= 0.02:
        print(f"  {name_b} does not change the dependence on the confounder.")
        print("  That refutes its rationale: a negative result backed by")
        print("  evidence, not a failure.")
    else:
        print(f"  The dependence RISES under {name_b}. The channel is being")
        print("  re-encoded instead of removed.")
    print("  This verdict covers the two endpoints above and NOTHING else.")
    print("  Grad-CAM, sensitivity gap and the perturbations live in")
    print("  results_rsna.csv and have to be read there before any variant")
    print("  is called the winner.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", type=Path, default=Path("predictions_rsna"))
    p.add_argument("--b", type=Path, default=Path("predictions_rsna_crop"))
    p.add_argument("--name-a", default="Basislinie")
    p.add_argument("--name-b", default="Zuschnitt")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--col", default="p_clean")
    p.add_argument("--out", type=Path,
                   default=Path("predictions_rsna/crop_compare.csv"))
    args = p.parse_args(argv)

    print(f"A = {args.a}  ({args.name_a})")
    print(f"B = {args.b}  ({args.name_b})")
    rows = compare(args.a, args.b, args.folds, args.seed, args.col)
    report(rows, args.name_a, args.name_b)
    if rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nsaved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
