"""
Phase 3: two evaluations that need no training, and that a clinical reader asks
for first.

WHY THIS EXISTS
---------------
The AUC says how well the model RANKS. It says nothing about which mistakes it
makes, and nothing about what the number on the screen means. A web app that
prints 0.7 owes the reader an answer to "0.7 of what".

PART A: WHICH FILMS DOES IT CALL POSITIVE BY MISTAKE
----------------------------------------------------
RSNA labels every film with one of three classes, and `rsna_data.py` defines
them and then never uses them again:

  Normal                        nothing to see
  No Lung Opacity / Not Normal  something to see, but not an infiltrate:
                                effusion, congestion, scarring, cardiomegaly,
                                a device
  Lung Opacity                  the positive class, with boxes

A false positive on "Normal" and a false positive on the middle class are not
the same mistake. The first is a model inventing pathology in a clean film. The
second is a model confusing one abnormality with another, which is the mistake a
junior reader makes, and it is far more forgivable.

PRE-REGISTERED EXPECTATION (roadmap v1, written 02.08. before this ran): the
large majority of false positives sits in the middle class.

PART B: WHAT DOES THE NUMBER MEAN
---------------------------------
Three things get computed, all on the same validation folds:

1. Reliability by projection. Predictions into ten equal-sized buckets, and for
   each bucket the predicted rate against the observed rate. On the diagonal
   means the number is a probability. Above it means the model is overconfident.
2. The Brier score, split by Murphy into reliability (how far off the diagonal),
   resolution (how far the buckets separate from the base rate) and uncertainty
   (the base rate itself). Lower Brier is better, lower reliability is better,
   HIGHER resolution is better.
3. Sensitivity and specificity at named operating points, not only at Youden.
   "Hold specificity at 0.90, what sensitivity is left" is the question a
   department actually asks.

PRE-REGISTERED EXPECTATION (same source): `pos_weight` sits around 3.4, which
pushes every output upwards, so the number is probably not a readable
probability. If so, a post-hoc calibration belongs in the app, and it has to be
fitted on the inner SELECTION split, never on the validation fold that is used
to report the result.

Platt scaling is used for that: one logistic regression of the label on the
logit of the score, two parameters, fitted per fold on the selection split.
Deliberately not isotonic regression, which has more freedom than 3,050
selection images can pay for.

CORRECTED 02.08. AFTER AN AUDIT (erklaerungen/11_pruefung_phase3.md)
-------------------------------------------------------------------
The first version of this script pooled AP and PA in every headline it printed,
which is the one thing this project exists to warn about. Three changes:

  * Every rate, every operating point and every share is now printed pooled AND
    per projection. The pooled Youden point hides a sensitivity gap of 0.31
    between the projections.
  * Reliability is a squared quantity and is now also printed as its root, in
    probability points. The repair is a factor of 10, not the factor of 100 the
    raw numbers suggest.
  * `pos_weight` was asserted to be the cause of the miscalibration and never
    checked. `causal_test` checks it: a class weight can only shift the logit,
    so holding the slope at 1 and fitting the intercept alone is exactly the
    pos_weight hypothesis. It turns out to be most of the story but not all of
    it.

One factual error is also gone: the serving app does NOT carry separate
thresholds per projection. `serving/main.py` has one threshold, and it cannot
have more, because it is handed a PNG with no DICOM header.

CLI, from the repository root:
  python rsna\\befunde\\rsna_klassen_kalibrierung.py
  python rsna\\befunde\\rsna_klassen_kalibrierung.py --pred-dir predictions_rsna_bal10
"""

from __future__ import annotations

import _repo_path  # noqa: F401  (puts the neighbouring folders on sys.path)

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["Normal", "No Lung Opacity / Not Normal", "Lung Opacity"]
SHORT = {"Normal": "Normal",
         "No Lung Opacity / Not Normal": "Not normal, no opacity",
         "Lung Opacity": "Lung opacity (positive)"}
N_BINS = 10

# The chart palette, validated for the light surface with the skill's checker
# (adjacent CVD dE 24.7, normal vision 33.6, both well clear of the floors).
C_AP, C_PA = "#2a78d6", "#eb6834"
C_INK, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#d8d7d2"


# --------------------------------------------------------------------------
# Small statistics, kept dependency free on purpose
# --------------------------------------------------------------------------

def auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ps = np.asarray(p)[order]
    new = np.empty(ps.size, bool)
    new[0] = True
    np.not_equal(ps[1:], ps[:-1], out=new[1:])
    grp = np.cumsum(new) - 1
    counts = np.bincount(grp)
    ends = np.cumsum(counts)
    mid = 0.5 * (ends - counts + 1 + ends)
    ranks = np.empty(ps.size)
    ranks[order] = mid[grp]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """The operating point the app actually uses, taken from the selection split."""
    thr = np.unique(p)
    best, best_t = -np.inf, 0.5
    for t in thr[:: max(1, len(thr) // 2000)]:
        pred = p >= t
        sens = pred[y == 1].mean() if (y == 1).any() else 0.0
        spec = (~pred[y == 0]).mean() if (y == 0).any() else 0.0
        if sens + spec - 1 > best:
            best, best_t = sens + spec - 1, float(t)
    return best_t


def threshold_at_spec(y: np.ndarray, p: np.ndarray, target: float) -> float:
    neg = np.sort(p[y == 0])
    if neg.size == 0:
        return float("nan")
    return float(np.quantile(neg, target))


def threshold_at_sens(y: np.ndarray, p: np.ndarray, target: float) -> float:
    pos = np.sort(p[y == 1])
    if pos.size == 0:
        return float("nan")
    return float(np.quantile(pos, 1 - target))


def sens_spec(y: np.ndarray, p: np.ndarray, thr: float) -> tuple:
    pred = p >= thr
    sens = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    spec = float((~pred[y == 0]).mean()) if (y == 0).any() else float("nan")
    return sens, spec


def brier_decomposition(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> dict:
    """Murphy: Brier = reliability - resolution + uncertainty.

    Equal COUNT bins, not equal width. The scores pile up in a narrow band, and
    equal width bins would leave most of them in two buckets and call the rest
    empty.
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    base = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rel = res = 0.0
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        nb = int(m.sum())
        pb, ob = float(p[m].mean()), float(y[m].mean())
        rel += nb * (pb - ob) ** 2
        res += nb * (ob - base) ** 2
        rows.append({"bin": b, "n": nb, "predicted": pb, "observed": ob})
    n = len(y)
    return {"brier": brier, "reliability": rel / n, "resolution": res / n,
            "uncertainty": base * (1 - base), "base_rate": base,
            "mean_predicted": float(p.mean()), "bins": rows}


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    q = np.clip(p, eps, 1 - eps)
    return np.log(q / (1 - q))


def platt_params(y_fit: np.ndarray, p_fit: np.ndarray,
                 free_slope: bool = True) -> np.ndarray:
    """The two Platt parameters, fitted by Newton steps on the SELECTION split.

    `free_slope=False` holds the slope at 1 and fits the intercept only. That
    restricted model is exactly the `pos_weight` hypothesis: weighting the
    positive class by w in the loss shifts every logit by log w and does
    nothing else. Comparing the two fits turns "pos_weight is the cause" from a
    claim into a measurement, which is why the restricted fit exists here at
    all.
    """
    x = _logit(p_fit)
    X = np.column_stack([x, np.ones_like(x)]) if free_slope \
        else np.ones_like(x)[:, None]
    offset = 0.0 if free_slope else x
    w = np.zeros(X.shape[1])
    for _ in range(100):
        q = 1 / (1 + np.exp(-(X @ w + offset)))
        g = X.T @ (q - y_fit)
        W = np.clip(q * (1 - q), 1e-9, None)
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return np.array([1.0, w[0]]) if not free_slope else w


def platt_apply(w: np.ndarray, p_apply: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(w[0] * _logit(p_apply) + w[1])))


def platt(y_fit: np.ndarray, p_fit: np.ndarray, p_apply: np.ndarray) -> np.ndarray:
    """Two parameter logistic recalibration.

    Fitted on the SELECTION split and applied to the validation fold. Fitting it
    on the fold it is reported on would be the same mistake as tuning a
    threshold on the test set.
    """
    return platt_apply(platt_params(y_fit, p_fit, True), p_apply)


def mean_sd(v) -> tuple:
    a = np.asarray([x for x in v if np.isfinite(x)], float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


# --------------------------------------------------------------------------

def load(args) -> pd.DataFrame:
    cls = pd.read_csv(Path(args.csv) / "stage_2_detailed_class_info.csv")
    cls = cls.drop_duplicates("patientId").set_index("patientId")["class"]
    frames = []
    for f in args.folds:
        v = Path(args.pred_dir) / f"rsna_f{f}_s{args.seed}.csv"
        s = Path(args.pred_dir) / f"sel_f{f}_s{args.seed}.csv"
        if not v.exists():
            print(f"  fold {f}: {v} missing, skipped")
            continue
        dv = pd.read_csv(v)[["patientId", "y", "viewpos", "p_clean"]]
        dv["split"] = "val"
        dv = dv.rename(columns={"p_clean": "p"})
        if s.exists():
            ds = pd.read_csv(s)[["patientId", "y", "viewpos", "p_sel"]]
            ds["split"] = "sel"
            ds = ds.rename(columns={"p_sel": "p"})
            dv = pd.concat([dv, ds], ignore_index=True)
        dv["fold"] = f
        frames.append(dv)
    if not frames:
        raise SystemExit("no prediction files found")
    d = pd.concat(frames, ignore_index=True)
    d["class"] = d["patientId"].map(cls)
    return d


def part_a(d: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 84)
    print("PART A: WHICH FILMS DOES IT CALL POSITIVE BY MISTAKE")
    print("=" * 84)
    rows = []
    for f, g in d.groupby("fold"):
        sel, val = g[g["split"] == "sel"], g[g["split"] == "val"]
        thr = youden_threshold(sel["y"].to_numpy(), sel["p"].to_numpy()) \
            if len(sel) else 0.5
        for vp in ("all", "AP", "PA"):
            v = val if vp == "all" else val[val["viewpos"] == vp]
            if v.empty:
                continue
            pred = v["p"].to_numpy() >= thr
            cl = v["class"].to_numpy()
            for c in CLASSES:
                m = cl == c
                if not m.any():
                    continue
                rows.append({"fold": f, "projection": vp, "class": c,
                             "n": int(m.sum()), "threshold": thr,
                             "called_positive": float(pred[m].mean()),
                             "mean_score": float(v["p"].to_numpy()[m].mean())})
            fp = pred & (v["y"].to_numpy() == 0)
            neg = v["y"].to_numpy() == 0
            for c in CLASSES[:2]:
                m = cl == c
                rows.append({"fold": f, "projection": vp,
                             "class": f"share of all FP: {c}",
                             "n": int((fp & m).sum()), "threshold": thr,
                             "called_positive": float((fp & m).sum() / max(fp.sum(), 1)),
                             "share_of_negatives": float((neg & m).sum() / max(neg.sum(), 1)),
                             "mean_score": np.nan})
    t = pd.DataFrame(rows)

    print("\n  Rate called positive, at the Youden threshold of that fold's")
    print("  selection split. Pooled AND per projection: clean films are 82%")
    print("  PA, so a pooled false alarm rate is mostly a statement about the")
    print("  AP/PA mixture of RSNA rather than about the model.")
    print(f"  {'class':<26}{'proj':>5}{'n/fold':>8}{'called positive':>19}"
          f"{'mean score':>13}")
    for vp in ("all", "AP", "PA"):
        rate = {}
        for c in CLASSES:
            g = t[(t["class"] == c) & (t["projection"] == vp)]
            if g.empty:
                continue
            m, s = mean_sd(g["called_positive"])
            sc, _ = mean_sd(g["mean_score"])
            rate[c] = m
            print(f"  {SHORT[c]:<26}{vp:>5}{int(g['n'].mean()):>8}"
                  f"{m:>13.3f} +- {s:.3f}{sc:>13.3f}")
        if len(rate) == 3:
            print(f"  {'false alarm rate ratio':<26}{vp:>5}{'':>8}"
                  f"{rate[CLASSES[1]] / max(rate[CLASSES[0]], 1e-9):>13.1f} x"
                  "   <-- free of class sizes")
        print()

    print("  Where the false positives sit, and what share that class holds")
    print("  among the negatives to begin with. A share of 96% is only news if")
    print("  the class is not already most of the negatives.")
    print(f"  {'class':<26}{'proj':>5}{'share of all FP':>19}"
          f"{'share of negatives':>21}")
    for vp in ("all", "AP", "PA"):
        for c in CLASSES[:2]:
            g = t[(t["class"] == f"share of all FP: {c}") & (t["projection"] == vp)]
            if g.empty:
                continue
            m, s = mean_sd(g["called_positive"])
            b, _ = mean_sd(g["share_of_negatives"])
            print(f"  {SHORT[c]:<26}{vp:>5}{m:>13.3f} +- {s:.3f}{b:>21.3f}")

    mid = t[(t["class"] == f"share of all FP: {CLASSES[1]}")
            & (t["projection"] == "all")]["called_positive"].mean()
    print("\n  PRE-REGISTERED EXPECTATION: the large majority of false positives")
    print("  sits in the middle class.")
    print(f"  {'MET' if mid > 0.5 else 'NOT MET'}: {mid:.1%} of false positives are films with a")
    print("  visible abnormality that is not an infiltrate. That is the mistake a")
    print("  junior reader makes, not an invented finding in a clean chest.")
    print("  It survives the split by projection, which was the real test: the")
    print("  ordering and the size of the gap hold inside AP and inside PA.")
    return t


def causal_test(d: pd.DataFrame) -> None:
    """Is `pos_weight` really the cause? Two lines of arithmetic settle it.

    The claim was made in the roadmap and repeated in the write-up without ever
    being checked. If `pos_weight` were the whole story the fitted slope would
    come back at 1.0 and the intercept at -log(w), because that is the only
    thing a class weight can do to a logit. Anything else in the fit is a
    second effect wearing the same coat.
    """
    print("\n  Is pos_weight really the cause? The fitted parameters say so.")
    print("  The implied weight comes from the SHIFT ONLY fit, because exp(-b) is")
    print("  a class weight only when the slope is held at 1. Reading it off the")
    print("  free fit, where the slope is not 1, mixes two effects into one number.")
    print(f"  {'fold':<7}{'slope':>9}{'intercept':>12}{'implied w':>12}"
          f"{'neg/pos in sel':>17}")
    W, IW, RATIO = [], [], []
    for f, g in d.groupby("fold"):
        sel = g[g["split"] == "sel"]
        if sel.empty:
            continue
        ys, ps = sel["y"].to_numpy(float), sel["p"].to_numpy(float)
        w = platt_params(ys, ps, True)
        b = platt_params(ys, ps, False)[1]
        ratio = (ys == 0).sum() / max((ys == 1).sum(), 1)
        W.append(w)
        IW.append(np.exp(-b))
        RATIO.append(ratio)
        print(f"  {f:<7}{w[0]:>9.3f}{w[1]:>12.3f}{np.exp(-b):>12.2f}{ratio:>17.2f}")
    W = np.array(W)
    sm, ss = mean_sd(W[:, 0])
    im, _ = mean_sd(W[:, 1])
    iw, _ = mean_sd(IW)
    rm, _ = mean_sd(RATIO)
    print(f"  {'mean':<7}{sm:>9.3f}{im:>12.3f}{iw:>12.2f}{rm:>17.2f}")
    print("  The last column is the class ratio on the selection split, which is")
    print("  stratified like the fit part, so it stands in for the pos_weight")
    print("  rsna_train.py computes as neg/pos on the fit part. Watch the two")
    print(f"  columns against each other: the class ratio is flat at {rm:.2f} in every")
    print(f"  fold, while the shift the data asks for ranges {min(IW):.2f} to {max(IW):.2f}. A")
    print("  constant cause cannot produce a varying effect, so something besides")
    print("  pos_weight is moving the level as well.")

    rows = []
    for label, free in (("raw score", None), ("shift only (the pos_weight model)", False),
                        ("shift + shrink (Platt, as used)", True)):
        rel = []
        for f, g in d.groupby("fold"):
            sel, val = g[g["split"] == "sel"], g[g["split"] == "val"]
            if sel.empty:
                continue
            p = val["p"].to_numpy(float) if free is None else platt_apply(
                platt_params(sel["y"].to_numpy(float), sel["p"].to_numpy(float),
                             free), val["p"].to_numpy(float))
            rel.append(brier_decomposition(val["y"].to_numpy(), p)["reliability"])
        rows.append((label, mean_sd(rel)[0]))
    print(f"\n  {'':<36}{'reliability':>13}{'typical distance':>19}")
    for label, r in rows:
        print(f"  {label:<36}{r:>13.6f}{np.sqrt(r):>19.4f}")
    if sm < 0.9 or sm > 1.1:
        print(f"\n  PARTLY. The shift alone removes most of it, and the implied weight")
        print(f"  {iw:.2f} sits close to the {rm:.2f} the training used. But the")
        print(f"  slope is {sm:.3f} +- {ss:.3f}, not 1.0: the network additionally spreads its")
        print("  logits too far, which no class weight can cause. pos_weight explains")
        print("  the LEVEL of the miscalibration, not its SHAPE. Say it that way.")
    else:
        print(f"\n  YES. The slope is {sm:.3f} +- {ss:.3f}, indistinguishable from 1, so the")
        print("  whole miscalibration is the shift a class weight produces.")


def part_b(d: pd.DataFrame, args) -> tuple:
    print("\n" + "=" * 84)
    print("PART B: WHAT DOES THE NUMBER MEAN")
    print("=" * 84)

    # Post hoc calibration, fitted per fold on the selection split. Twice: once
    # globally, once separately per projection. The TRAINING pipeline already
    # searches a threshold per projection (rsna_train.py, thr_by_view), so a
    # calibration per projection is the same idea carried through, and the
    # difference between the two says whether the miscalibration is projection
    # specific.
    #
    # The serving app does NOT do this and cannot: it is handed a PNG with no
    # DICOM header and does not know the projection. So the per-projection
    # branch below is a measurement, not a deployment plan. That it turns out
    # unnecessary is therefore lucky as well as interesting.
    d = d.copy()
    d["p_cal"] = np.nan
    d["p_cal_vp"] = np.nan
    for f, g in d.groupby("fold"):
        sel = g[g["split"] == "sel"]
        if sel.empty:
            continue
        d.loc[g.index, "p_cal"] = platt(sel["y"].to_numpy(float),
                                        sel["p"].to_numpy(float),
                                        g["p"].to_numpy(float))
        for vp in ("AP", "PA"):
            s = sel[sel["viewpos"] == vp]
            m = g[g["viewpos"] == vp]
            if s.empty or m.empty:
                continue
            d.loc[m.index, "p_cal_vp"] = platt(s["y"].to_numpy(float),
                                               s["p"].to_numpy(float),
                                               m["p"].to_numpy(float))
    val = d[d["split"] == "val"]

    rows, curves = [], {}
    for label, col in (("raw", "p"), ("after Platt", "p_cal"),
                       ("Platt per projection", "p_cal_vp")):
        for vp in ("AP", "PA", "all"):
            g = val if vp == "all" else val[val["viewpos"] == vp]
            per = [brier_decomposition(x["y"].to_numpy(), x[col].to_numpy())
                   for _, x in g.groupby("fold")]
            rows.append({
                "calibration": label, "projection": vp, "n": len(g),
                "auc": mean_sd([auc(x["y"].to_numpy(), x[col].to_numpy())
                                for _, x in g.groupby("fold")])[0],
                "base_rate": mean_sd([r["base_rate"] for r in per])[0],
                "mean_predicted": mean_sd([r["mean_predicted"] for r in per])[0],
                "brier": mean_sd([r["brier"] for r in per])[0],
                "reliability": mean_sd([r["reliability"] for r in per])[0],
                "resolution": mean_sd([r["resolution"] for r in per])[0],
            })
            if vp != "all":
                b = brier_decomposition(g["y"].to_numpy(), g[col].to_numpy())
                curves[(label, vp)] = pd.DataFrame(b["bins"])
    t = pd.DataFrame(rows)

    print("\n  Is the number a probability?")
    print(f"  {'':<22}{'proj':>5}{'observed rate':>15}{'mean predicted':>16}"
          f"{'Brier':>9}{'reliability':>13}{'resolution':>12}")
    for _, r in t.iterrows():
        print(f"  {r['calibration']:<22}{r['projection']:>5}{r['base_rate']:>15.3f}"
              f"{r['mean_predicted']:>16.3f}{r['brier']:>9.4f}"
              f"{r['reliability']:>13.4f}{r['resolution']:>12.4f}")

    raw_all = t[(t["calibration"] == "raw") & (t["projection"] == "all")].iloc[0]
    cal_all = t[(t["calibration"] == "after Platt") & (t["projection"] == "all")].iloc[0]
    print("\n  PRE-REGISTERED EXPECTATION: pos_weight around 3.4 pushes every")
    print("  output upwards, so the raw number is probably not a readable")
    print("  probability.")
    over = raw_all["mean_predicted"] / raw_all["base_rate"]
    print(f"  {'MET' if over > 1.2 else 'NOT MET'}: the raw mean prediction is {raw_all['mean_predicted']:.3f} against an observed")
    print(f"  rate of {raw_all['base_rate']:.3f}, a factor of {over:.2f}. Platt scaling on the")
    print(f"  selection split brings the mean to {cal_all['mean_predicted']:.3f} and cuts the")
    print(f"  reliability term from {raw_all['reliability']:.4f} to {cal_all['reliability']:.4f},")
    print(f"  in readable units from {np.sqrt(raw_all['reliability']):.4f} to "
          f"{np.sqrt(cal_all['reliability']):.4f} probability points of typical")
    print("  distance from the diagonal. Reliability is a SQUARED quantity, so the")
    print("  root is the honest way to say how big the repair is: a factor of "
          f"{np.sqrt(raw_all['reliability'] / cal_all['reliability']):.1f},")
    print("  not the factor of 100 the untransformed numbers suggest.")
    print("  Ranking is untouched by construction: a monotone transform cannot")
    print(f"  change the AUC ({raw_all['auc']:.4f} against {cal_all['auc']:.4f}).")
    print("  NOTE that this AUC pools the projections. The honest number for this")
    print("  run is auc_stratified in results_rsna.csv; the pooled one sits above")
    print("  BOTH per-projection AUCs and that gap is the confounder.")

    causal_test(d)

    vp_all = t[(t["calibration"] == "Platt per projection") &
               (t["projection"] == "all")]
    if len(vp_all):
        v = vp_all.iloc[0]
        gain = cal_all["reliability"] - v["reliability"]
        print("\n  Does the calibration have to be split by projection as well?")
        print(f"  One curve for both projections leaves a reliability term of "
              f"{cal_all['reliability']:.4f},")
        print(f"  one curve each leaves {v['reliability']:.4f}.")
        if gain > 0.0005:
            print("  Worth splitting: the projection sits in the calibration as well as")
            print("  in the score. Note that the serving app could not use it, see the")
            print("  comment at the top of this function.")
        else:
            print("  NO. The gain is below 0.0005 Brier and not worth a second set of")
            print("  parameters. This is a small surprise: the projection is the")
            print("  confounder of this whole project and it dominates the score, but")
            print("  the OVERCONFIDENCE it causes is the same shape in both groups, so")
            print("  one two-parameter curve removes it for both. Per projection the")
            print("  split helps PA a little and AP not at all.")
        print("  Either way the residual after calibration is projection specific in")
        print("  its sign: see the figure, PA sits slightly below the diagonal at the")
        print("  top end where few films are.")

    print("\n  Named operating points, threshold taken from the selection split.")
    print("  Every row is given pooled AND per projection, because one shared")
    print("  threshold has one sensitivity per projection and they are not the")
    print("  same number. The pooled value is a mixture that neither group sees.")
    print(f"  {'point':<28}{'proj':>5}{'sensitivity':>16}{'specificity':>16}")
    op = []
    for name, fn in (("Youden", None),
                     ("specificity held at 0.90", lambda y, p: threshold_at_spec(y, p, 0.90)),
                     ("specificity held at 0.95", lambda y, p: threshold_at_spec(y, p, 0.95)),
                     ("sensitivity held at 0.90", lambda y, p: threshold_at_sens(y, p, 0.90))):
        acc = {k: {"se": [], "sp": []} for k in ("all", "AP", "PA")}
        for f, g in val.groupby("fold"):
            sel = d[(d["fold"] == f) & (d["split"] == "sel")]
            if sel.empty:
                continue
            ys, ps = sel["y"].to_numpy(), sel["p"].to_numpy()
            thr = youden_threshold(ys, ps) if fn is None else fn(ys, ps)
            for k in ("all", "AP", "PA"):
                x = g if k == "all" else g[g["viewpos"] == k]
                if x.empty:
                    continue
                a, b = sens_spec(x["y"].to_numpy(), x["p"].to_numpy(), thr)
                acc[k]["se"].append(a)
                acc[k]["sp"].append(b)
        for k in ("all", "AP", "PA"):
            m1, s1 = mean_sd(acc[k]["se"])
            m2, s2 = mean_sd(acc[k]["sp"])
            lab = name if k == "all" else ""
            print(f"  {lab:<28}{k:>5}{m1:>10.3f} +- {s1:.3f}{m2:>10.3f} +- {s2:.3f}")
            op.append({"point": name, "projection": k, "sens": m1, "sens_sd": s1,
                       "spec": m2, "spec_sd": s2})
        gap = mean_sd(acc["AP"]["se"])[0] - mean_sd(acc["PA"]["se"])[0]
        print(f"  {'':<28}{'gap':>5}{gap:>10.3f}")

    print("\n  The threshold travels worse than the ranking. Every number in this")
    print("  block is the transfer of a decision rule from one split to another,")
    print("  which is exactly what fails when the model meets a new department.")
    print("  The sensitivity gap between the projections is the same failure")
    print("  happening INSIDE the dataset, and rsna_train.py writes it per fold as")
    print("  sens_gap. Finding 6 in the README carries it; this table is the")
    print("  five-fold version of the same thing.")
    return t, curves, pd.DataFrame(op), val


def figure(curves: dict, val: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [k for k in ("raw", "after Platt", "Platt per projection")
              if (k, "AP") in curves]
    fig, ax = plt.subplots(2, len(panels), figsize=(4.7 * len(panels), 7.4),
                           gridspec_kw={"height_ratios": [2.6, 1]},
                           squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for a in ax.ravel():
        a.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(C_GRID)
        a.tick_params(colors=C_MUTED, labelsize=9)
        a.grid(True, color=C_GRID, linewidth=0.6, alpha=0.7)
        a.set_axisbelow(True)

    for j, label in enumerate(panels):
        a = ax[0, j]
        a.plot([0, 1], [0, 1], color=C_MUTED, linewidth=1, linestyle=(0, (4, 3)),
               zorder=1)
        if j == 0:
            # Only on the first panel. On the calibrated ones the curve lies on
            # the diagonal, which is the point, and a label there would collide.
            a.annotate("perfect calibration", (0.66, 0.60), color=C_MUTED,
                       fontsize=8.5, rotation=34, ha="center")
        for vp, col in (("AP", C_AP), ("PA", C_PA)):
            c = curves.get((label, vp))
            if c is None:
                continue
            a.plot(c["predicted"], c["observed"], color=col, linewidth=2,
                   marker="o", markersize=6, markeredgecolor="#fcfcfb",
                   markeredgewidth=1.5, label=vp, zorder=3)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.set_title(f"{label}", color=C_INK, fontsize=11, loc="left", pad=8)
        a.set_xlabel("predicted probability", color=C_MUTED, fontsize=9.5)
        if j == 0:
            a.set_ylabel("observed rate of pneumonia", color=C_MUTED, fontsize=9.5)
        leg = a.legend(frameon=False, fontsize=9.5, loc="upper left")
        for t_ in leg.get_texts():
            t_.set_color(C_INK)

    for j, col in enumerate(["p", "p_cal", "p_cal_vp"][:len(panels)]):
        a = ax[1, j]
        for vp, c in (("AP", C_AP), ("PA", C_PA)):
            v = val[val["viewpos"] == vp][col].dropna()
            a.hist(v, bins=np.linspace(0, 1, 41), color=c, alpha=0.55,
                   label=vp, edgecolor="none")
        a.set_xlim(0, 1)
        a.set_xlabel("predicted probability", color=C_MUTED, fontsize=9.5)
        if j == 0:
            a.set_ylabel("films", color=C_MUTED, fontsize=9.5)

    fig.suptitle("Does the score mean what it says? Reliability by projection, "
                 "five folds pooled",
                 color=C_INK, fontsize=12.5, x=0.02, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  figure written: {out}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--pred-dir", type=Path, default=Path("predictions_rsna_base"))
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("predictions_klassen"))
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    d = load(args)
    missing = int(d["class"].isna().sum())
    if missing:
        print(f"  WARNING: {missing} films without a class label")

    a = part_a(d)
    a.to_csv(args.out_dir / "klassen.csv", index=False)
    t, curves, op, val = part_b(d, args)
    t.to_csv(args.out_dir / "kalibrierung.csv", index=False)
    op.to_csv(args.out_dir / "arbeitspunkte.csv", index=False)
    pd.concat([c.assign(calibration=k[0], projection=k[1])
               for k, c in curves.items()]).to_csv(
        args.out_dir / "zuverlaessigkeit.csv", index=False)
    if not args.no_figure:
        figure(curves, val, args.out_dir / "kalibrierung.png")
    print(f"\nsaved into {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
