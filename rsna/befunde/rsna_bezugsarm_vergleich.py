"""Does the hardware change the result? The reference arm, APU against card.

WHY THIS EXISTS
---------------
Phase 4 moved the training from adapter 0 (the integrated graphics) to adapter 1
(the RX 5500 XT). The roadmap rule that follows is hard: both arms of a
comparison must run on the same chip, so the 224 px reference arm has to be
produced again on the card before phase 5 can compare anything against it.

That rerun is a cost. It is also an opportunity, because it answers a question
this project has so far only argued about: **does the choice of chip move the
result?** Two five-fold runs, identical in every switch, differing only in the
adapter, is exactly the experiment for it.

Phase 4 already showed that ONE forward pass agrees to float32 rounding
(2.4e-06). That says nothing about a trained model. Eight epochs of rounding
differences produce different weights, and the honest expectation is that the
two arms differ image by image. The question is whether they differ in the
SUMMARY.

PRE-REGISTERED BEFORE THE RERUN EXISTS
--------------------------------------
Endpoint A, stratified AUC, n-weighted mean of the AP-only and PA-only AUC.
Same definition as `rsna_train.stratified_scores`, copied here on purpose so a
disagreement is a disagreement and not a shared bug.

Endpoint C, AUC(score -> ViewPosition), the primary endpoint of the project.

The test is an EQUIVALENCE test, not the usual one, and that distinction is the
whole point. "No significant difference" would be worthless here: with five
folds almost nothing is significant, so a sloppy test would rubber-stamp any
hardware. Asked instead: **is the difference small enough to be irrelevant?**

  margin  delta = 0.01 stratified AUC. Not invented for this run: it is the
          same size as the minimum difference the roadmap pre-registered for
          the resolution comparison in phase 8. A hardware effect smaller than
          the smallest effect the project is willing to act on cannot change
          any decision.
  method  paired per fold (cross-over: same fold, both chips), 90 percent
          confidence interval of the mean paired difference, TOST. Equivalence
          holds when the whole interval lies inside [-delta, +delta].
  verdict PASS      interval inside the margin. The arms are interchangeable
                    at the level the project cares about.
          FAIL      interval entirely outside. The hardware moved the result,
                    which is a finding and stops phase 5 until it is understood.
          UNCLEAR   interval straddles the margin. NOT a pass. It means the
                    five folds cannot separate the two, and the honest reading
                    is "too imprecisely measured", not "no difference".

Reported alongside, because a summary that agrees can still hide two different
models: the largest per-image difference in the predicted probability, and the
rank correlation of the two arms per fold.

Nothing here trains. Input is the two prediction directories. Runtime seconds.

CLI, from the repository root:
  python rsna\\befunde\\rsna_bezugsarm_vergleich.py
  python rsna\\befunde\\rsna_bezugsarm_vergleich.py --karte predictions_rsna_dml1
"""

from __future__ import annotations

import _repo_path  # noqa: F401

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

DELTA = 0.01          # equivalence margin on the stratified AUC
DELTA_C = 0.02        # and on endpoint C, twice as wide: C is a diagnostic
                      # number, not a claim about patients
ALPHA = 0.10          # 90 % interval, the standard choice for TOST at 5 %

# Phase 0, five folds on adapter 0. Kept here as the historical reference.
APU_A_MEAN, APU_A_SD = 0.8449, 0.0147
APU_C_MEAN, APU_C_SD = 0.8166, 0.0098

FINDINGS: list[str] = []


def check(name: str, cond: bool, info: str = "") -> bool:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FINDINGS.append(name)
    return cond


def note(text: str) -> None:
    print(f"        {text}")


# --------------------------------------------------------------------------

def stratified_auc(y: np.ndarray, p: np.ndarray, vp: np.ndarray) -> float:
    """n-weighted mean of the AP-only and PA-only AUC.

    The overall AUC still contains the projection effect, which is the whole
    confounder of this project. Only this number says something about
    radiology.
    """
    parts, weights = [], []
    for v in ("AP", "PA"):
        m = vp == v
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        parts.append(roc_auc_score(y[m], p[m]))
        weights.append(m.sum())
    if len(parts) < 2:
        return float("nan")
    w = np.asarray(weights, float)
    return float(np.dot(parts, w) / w.sum())


def score_to_view(p: np.ndarray, vp: np.ndarray) -> float:
    """Endpoint C. Deliberately not folded to max(a, 1 - a): the direction
    carries meaning and 0.5 is the floor of the channel, not of the number."""
    m = np.isin(vp, ["AP", "PA"])
    if not m.sum() or len(set(vp[m])) < 2:
        return float("nan")
    return float(roc_auc_score((vp[m] == "AP").astype(int), p[m]))


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def load(pred_dir: Path, fold: int, seed: int) -> pd.DataFrame:
    f = Path(pred_dir) / f"rsna_f{fold}_s{seed}.csv"
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f).set_index("patientId").sort_index()


def tost(diffs: np.ndarray, delta: float, label: str) -> str:
    """Paired equivalence test. Returns PASS, FAIL or UNCLEAR."""
    n = len(diffs)
    m = float(np.mean(diffs))
    if n < 2:
        print(f"  {label}: one fold only, no interval, NO VERDICT")
        FINDINGS.append(f"{label}: fewer than two folds")
        return "NO VERDICT"
    sd = float(np.std(diffs, ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(1 - ALPHA / 2, n - 1))
    lo, hi = m - tcrit * se, m + tcrit * se
    print(f"  {label}: mean difference {m:+.4f}, sd {sd:.4f}, "
          f"{int((1 - ALPHA) * 100)} % interval [{lo:+.4f}, {hi:+.4f}], "
          f"margin ±{delta:.3f}")

    if -delta < lo and hi < delta:
        verdict = "PASS"
    elif lo >= delta or hi <= -delta:
        verdict = "FAIL"
    else:
        verdict = "UNCLEAR"
    print(f"  {label}: {verdict}")

    if verdict == "PASS":
        note("The interval lies inside the margin. The chip does not move this")
        note("endpoint by anything the project would act on.")
    elif verdict == "FAIL":
        note("The interval lies entirely outside the margin. The hardware")
        note("moved the endpoint. This stops phase 5 until it is understood.")
        FINDINGS.append(f"{label}: the hardware moved the endpoint")
    else:
        note("The interval straddles the margin. This is NOT a pass and NOT a")
        note("refutation, it is 'measured too imprecisely'. Five folds cannot")
        note("separate the two. The remedy is more folds or more seeds, not a")
        note("shrug.")
        # The number that says how much precision is missing.
        mde = tcrit * se
        note(f"Half-width of the interval: {mde:.4f}. Anything smaller than")
        note("that cannot be resolved with this many folds.")
        FINDINGS.append(f"{label}: too imprecise for a verdict")
    return verdict


# --------------------------------------------------------------------------

def check_epoch_choice(a, d: pd.DataFrame) -> None:
    """Which fold diverges most, and is the epoch choice the reason?

    Added on 04.08. after the first real run. Fold 0 diverged image by image
    about twice as much as the other four, and the obvious suspicion was the
    checkpoint selection: the two arms may have kept a DIFFERENT epoch. Then
    they are not the same model observed twice, they are two different models,
    and the divergence has nothing to do with the chip.

    The suspicion is testable without any retraining, which is why it is a test
    and not a remark. Every prediction file carries `p_last_epoch` next to
    `p_clean`, and the last epoch is by definition the same epoch in both arms.
    Comparing those two columns holds the epoch fixed. If the epoch choice is
    the explanation, the outlier fold has to fall back in line there.
    """
    print("\n4  IS THE CHECKPOINT CHOICE THE REASON FOR THE DIVERGENCE")
    rows = []
    for f in a.folds:
        ha = Path(a.apu) / f"history_f{f}_s{a.seed}.csv"
        hb = Path(a.karte) / f"history_f{f}_s{a.seed}.csv"
        if not (ha.exists() and hb.exists()):
            print(f"  fold {f}: no history file, skipped")
            continue
        A, B = pd.read_csv(ha), pd.read_csv(hb)
        ea = int(A.loc[A.is_best == 1, "epoch"].max())
        eb = int(B.loc[B.is_best == 1, "epoch"].max())
        # How close was the decision? A checkpoint picked on a margin of 0.0003
        # is a coin toss, and a rounding difference is enough to flip it.
        sa = np.sort(A["sel_auc"].to_numpy())[::-1]
        sb = np.sort(B["sel_auc"].to_numpy())[::-1]
        rows.append({"fold": f, "epoch_apu": ea, "epoch_karte": eb,
                     "same": ea == eb,
                     "margin_apu": float(sa[0] - sa[1]),
                     "margin_karte": float(sb[0] - sb[1])})
    if not rows:
        return
    e = pd.DataFrame(rows)
    print(f"  {'fold':>4} {'epoch APU':>10} {'epoch card':>11} {'same':>6}"
          f" {'margin APU':>11} {'margin card':>12}")
    for _, r in e.iterrows():
        print(f"  {int(r['fold']):>4} {int(r['epoch_apu']):>10} "
              f"{int(r['epoch_karte']):>11} {str(bool(r['same'])):>6} "
              f"{r['margin_apu']:>+11.4f} {r['margin_karte']:>+12.4f}")

    tight = e[e["margin_apu"] < 0.001]
    if len(tight):
        note(f"Folds picked on a margin below 0.001: "
             f"{sorted(int(x) for x in tight['fold'])}. At that distance the")
        note("choice is a coin toss and a rounding difference decides it.")

    mismatch = e[~e["same"].astype(bool)]
    if mismatch.empty:
        print("  ok    both arms kept the same epoch everywhere")
        note("So the per-image divergence above is the pure effect of two")
        note("trainings, not of two different checkpoints.")
        return

    print(f"  NOTE  {len(mismatch)} fold(s) kept a different epoch: "
          f"{sorted(int(x) for x in mismatch['fold'])}")

    # The decisive test: same epoch in both arms, does the outlier disappear?
    print("\n  Same comparison with the epoch held fixed (last epoch in both "
          "arms):")
    out = []
    for f in a.folds:
        A, B = load(a.apu, f, a.seed), load(a.karte, f, a.seed)
        if A.empty or B.empty or "p_last_epoch" not in A.columns:
            continue
        i = A.index.intersection(B.index)
        pa, pb = A.loc[i, "p_last_epoch"].to_numpy(), B.loc[i, "p_last_epoch"].to_numpy()
        sel_a = A.loc[i, "p_clean"].to_numpy()
        sel_b = B.loc[i, "p_clean"].to_numpy()
        out.append({"fold": f,
                    "mean_dp_selected": float(np.abs(sel_a - sel_b).mean()),
                    "mean_dp_fixed": float(np.abs(pa - pb).mean()),
                    "rank_fixed": rank_corr(pa, pb),
                    "same": bool(e.loc[e.fold == f, "same"].iloc[0])})
    if not out:
        return
    o = pd.DataFrame(out)
    print(f"  {'fold':>4} {'mean dp selected':>17} {'mean dp fixed':>14}"
          f" {'rank fixed':>11} {'same epoch':>11}")
    for _, r in o.iterrows():
        print(f"  {int(r['fold']):>4} {r['mean_dp_selected']:>17.4f} "
              f"{r['mean_dp_fixed']:>14.4f} {r['rank_fixed']:>11.4f} "
              f"{str(r['same']):>11}")

    # The verdict of this sub-test, written before the numbers: if the epoch
    # choice explains the outlier, the spread across folds must collapse once
    # the epoch is fixed.
    spread_sel = float(o["mean_dp_selected"].max() - o["mean_dp_selected"].min())
    spread_fix = float(o["mean_dp_fixed"].max() - o["mean_dp_fixed"].min())
    print(f"\n  spread across folds: {spread_sel:.4f} at the selected epoch, "
          f"{spread_fix:.4f} at a fixed epoch")
    if spread_fix < spread_sel / 2:
        print("  ok    the outlier disappears once the epoch is held fixed")
        note("So it was NOT the chip. The two arms kept different checkpoints,")
        note("and different checkpoints are different models. At a fixed epoch")
        note("all folds diverge by the same small amount, which is the honest")
        note("size of the pure hardware effect on a trained model.")
    else:
        print("  NOTE  the spread survives, so the epoch choice does not "
              "explain it")
        note("Something else makes that fold special. Worth finding before")
        note("phase 5 uses it as a reference.")
        FINDINGS.append("one fold diverges more and the epoch choice does not "
                        "explain it")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--apu", type=Path, default=Path("predictions_rsna_base"),
                   help="the arm from adapter 0, the integrated graphics")
    p.add_argument("--karte", type=Path, default=Path("predictions_rsna_dml1"),
                   help="the arm from adapter 1, the RX 5500 XT")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", type=Path, default=Path("results_rsna.csv"))
    a = p.parse_args()

    print("=" * 70)
    print("REFERENCE ARM: adapter 0 (APU) against adapter 1 (card)")
    print("=" * 70)

    # ---- provenance first. File names have lied in this project before -----
    print("\n0  WHERE DO THE NEW ROWS SAY THEY COME FROM")
    if a.results.exists():
        r = pd.read_csv(a.results)
        if "dml_index" not in r.columns:
            check("results_rsna.csv carries the provenance columns", False,
                  "no dml_index column")
            note("The file predates phase 4. The card run has to be the one")
            note("that writes it, otherwise there is nothing to check.")
        else:
            card = r[r["dml_index"] == 1]
            folds = sorted(set(int(f) for f in card["fold"]))
            # Not "exactly five rows": the run is restartable, so a fold that
            # was interrupted and repeated legitimately leaves two rows. What
            # must hold is that every fold is there and that none of them came
            # from the other chip.
            check("every fold has a row from adapter 1",
                  folds == sorted(a.folds),
                  f"folds {folds}, {len(card)} row(s) in total")
            if len(card):
                names = sorted(set(str(x).strip() for x in card["device_name"]))
                check("all of them name the card",
                      names and all("RX 5500" in n for n in names),
                      "; ".join(names))
            if len(card) > len(a.folds):
                note(f"{len(card)} rows for {len(a.folds)} folds: a fold was")
                note("run more than once. Allowed, the prediction CSV of the")
                note("last run is what counts, but worth knowing.")
    else:
        check(f"{a.results} exists", False)

    # ---- the two arms, fold by fold ---------------------------------------
    print("\n1  THE TWO ARMS, FOLD BY FOLD")
    rows = []
    for f in a.folds:
        A, B = load(a.apu, f, a.seed), load(a.karte, f, a.seed)
        if A.empty or B.empty:
            print(f"  fold {f}: missing ({'APU' if A.empty else ''}"
                  f"{' and ' if A.empty and B.empty else ''}"
                  f"{'card' if B.empty else ''})")
            FINDINGS.append(f"fold {f} is missing in one of the arms")
            continue
        common = A.index.intersection(B.index)
        if len(common) != len(A) or len(common) != len(B):
            FINDINGS.append(f"fold {f}: the two arms hold different images")
            print(f"  fold {f}: FAIL, different image sets "
                  f"({len(A)} against {len(B)}, {len(common)} shared)")
        A, B = A.loc[common], B.loc[common]
        y = A["y"].to_numpy()
        vp = A["viewpos"].to_numpy().astype(str)
        pa, pb = A["p_clean"].to_numpy(), B["p_clean"].to_numpy()
        rows.append({
            "fold": f, "n": len(common),
            "A_apu": stratified_auc(y, pa, vp),
            "A_karte": stratified_auc(y, pb, vp),
            "C_apu": score_to_view(pa, vp),
            "C_karte": score_to_view(pb, vp),
            "max_dp": float(np.abs(pa - pb).max()),
            "mean_dp": float(np.abs(pa - pb).mean()),
            "rank_corr": rank_corr(pa, pb),
        })
    if not rows:
        print("\nNothing to compare. Run the card arm first.")
        sys.exit(1)

    d = pd.DataFrame(rows)
    print(f"\n  {'fold':>4} {'n':>6} {'A APU':>8} {'A Karte':>8} {'diff':>8}"
          f" {'C APU':>8} {'C Karte':>8} {'diff':>8}")
    for _, r in d.iterrows():
        print(f"  {int(r['fold']):>4} {int(r['n']):>6} {r['A_apu']:>8.4f} "
              f"{r['A_karte']:>8.4f} {r['A_karte'] - r['A_apu']:>+8.4f} "
              f"{r['C_apu']:>8.4f} {r['C_karte']:>8.4f} "
              f"{r['C_karte'] - r['C_apu']:>+8.4f}")
    print(f"  {'mean':>4} {'':>6} {d['A_apu'].mean():>8.4f} "
          f"{d['A_karte'].mean():>8.4f} "
          f"{(d['A_karte'] - d['A_apu']).mean():>+8.4f} "
          f"{d['C_apu'].mean():>8.4f} {d['C_karte'].mean():>8.4f} "
          f"{(d['C_karte'] - d['C_apu']).mean():>+8.4f}")

    print(f"\n  Phase 0 on the APU reported A = {APU_A_MEAN:.4f} ± "
          f"{APU_A_SD:.4f} and C = {APU_C_MEAN:.4f} ± {APU_C_SD:.4f}.")
    note("Those ± are the SPREAD ACROSS FOLDS, not a confidence interval of")
    note("the mean. They say how much folds differ from each other, which is")
    note("why the test below is paired and does not use them as a gate.")

    # ---- the equivalence test ---------------------------------------------
    print("\n2  EQUIVALENCE, PAIRED PER FOLD")
    tost((d["A_karte"] - d["A_apu"]).to_numpy(), DELTA,
         "A, stratified AUC")
    print()
    tost((d["C_karte"] - d["C_apu"]).to_numpy(), DELTA_C,
         "C, AUC(score -> ViewPosition)")

    # ---- and the part a summary hides --------------------------------------
    print("\n3  THE SAME SUMMARY DOES NOT MEAN THE SAME MODEL")
    for _, r in d.iterrows():
        print(f"  fold {int(r['fold'])}: largest per-image difference in the "
              f"probability {r['max_dp']:.4f}, mean {r['mean_dp']:.4f}, "
              f"rank correlation {r['rank_corr']:.4f}")
    worst = float(d["max_dp"].max())
    print(f"\n  Largest single-image difference over all folds: {worst:.4f}.")
    if worst > 0.05:
        note("Two arms whose summary numbers agree can still disagree by a lot")
        note("on an individual film. That is the concrete reason both arms of")
        note("a comparison have to sit on one adapter: mixing them would put")
        note("this much noise into a difference the project wants to measure")
        note("at the level of 0.01.")
    else:
        note("Unexpectedly small. Worth a look: are the two directories really")
        note("two runs, or was one of them copied?")

    check_epoch_choice(a, d)

    print("\n" + "=" * 70)
    if FINDINGS:
        print(f"{len(FINDINGS)} FINDING(S):")
        for f in FINDINGS:
            print(f"  - {f}")
    else:
        print("No finding. The card arm is the valid reference for phase 5.")
    sys.exit(1 if FINDINGS else 0)


if __name__ == "__main__":
    main()
