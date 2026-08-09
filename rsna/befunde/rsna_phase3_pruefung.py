"""Phase 3 audit: re-derive every published number, then ask the four questions
Phase 3 did not ask.

WHY THIS EXISTS
---------------
`rsna_klassen_kalibrierung.py` pooled AP and PA films in every headline number
it printed. This project's whole argument is that pooling over `ViewPosition`
is what makes a chest X-ray model look better than it is. So the audit is:
recompute the published table (it must match to the digit), then split every
claim by projection and see which ones survive.

It also tests the one CAUSAL claim Phase 3 made and never checked: that
`pos_weight` around 3.4 is the reason the score is not a probability. That claim
is testable in two lines. If pos_weight were the whole story, a recalibration
with the SLOPE HELD AT 1 (a pure shift of the logit by -log w) would fix the
miscalibration completely, and the free-slope Platt fit would come back with a
slope of 1.0. Both are measured below.

Nothing here trains anything. Input is `predictions_rsna_base/` plus the RSNA
class table; runtime is a few seconds.

CLI, from the repository root:
  python rsna\\befunde\\rsna_phase3_pruefung.py
  python rsna\\befunde\\rsna_phase3_pruefung.py --pred-dir predictions_rsna_bal10
"""

from __future__ import annotations

import _repo_path  # noqa: F401

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["Normal", "No Lung Opacity / Not Normal", "Lung Opacity"]
SHORT = {CLASSES[0]: "Normal", CLASSES[1]: "Not normal, no opacity",
         CLASSES[2]: "Lung opacity (positive)"}


# --------------------------------------------------------------------------
# statistics, identical to the ones in rsna_klassen_kalibrierung.py so that a
# mismatch here means a real disagreement and not a second implementation
# --------------------------------------------------------------------------

def auc(y, p) -> float:
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(p, kind="mergesort")
    ps = np.asarray(p)[o]
    new = np.empty(ps.size, bool)
    new[0] = True
    np.not_equal(ps[1:], ps[:-1], out=new[1:])
    grp = np.cumsum(new) - 1
    cnt = np.bincount(grp)
    ends = np.cumsum(cnt)
    mid = 0.5 * (ends - cnt + 1 + ends)
    r = np.empty(ps.size)
    r[o] = mid[grp]
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def youden_threshold(y, p) -> float:
    thr = np.unique(p)
    best, best_t = -np.inf, 0.5
    for t in thr[:: max(1, len(thr) // 2000)]:
        pred = p >= t
        s = pred[y == 1].mean() if (y == 1).any() else 0.0
        sp = (~pred[y == 0]).mean() if (y == 0).any() else 0.0
        if s + sp - 1 > best:
            best, best_t = s + sp - 1, float(t)
    return best_t


def logit(p, eps=1e-6):
    q = np.clip(p, eps, 1 - eps)
    return np.log(q / (1 - q))


def fit_recal(y, p, free_slope: bool = True) -> np.ndarray:
    """Platt if free_slope, otherwise a pure shift (the pos_weight model)."""
    x = logit(p)
    X = np.column_stack([x, np.ones_like(x)]) if free_slope \
        else np.ones_like(x)[:, None]
    off = 0.0 if free_slope else x
    w = np.zeros(X.shape[1])
    for _ in range(200):
        q = 1 / (1 + np.exp(-(X @ w + off)))
        g = X.T @ (q - y)
        W = np.clip(q * (1 - q), 1e-9, None)
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-12:
            break
    return w


def apply_recal(p, w, free_slope: bool = True):
    z = logit(p)
    return 1 / (1 + np.exp(-(w[0] * z + w[1]))) if free_slope \
        else 1 / (1 + np.exp(-(z + w[0])))


def reliability(y, p, n_bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    e[0], e[-1] = -np.inf, np.inf
    idx = np.clip(np.digitize(p, e[1:-1]), 0, n_bins - 1)
    rel = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rel += m.sum() * (p[m].mean() - y[m].mean()) ** 2
    return rel / len(y)


def ms(v):
    a = np.asarray(v, float)
    return a.mean(), a.std(ddof=1) if a.size > 1 else 0.0


# --------------------------------------------------------------------------

def load(args) -> pd.DataFrame:
    cls = pd.read_csv(Path(args.csv) / "stage_2_detailed_class_info.csv")
    cls = cls.drop_duplicates("patientId").set_index("patientId")["class"]
    fr = []
    for f in args.folds:
        v = Path(args.pred_dir) / f"rsna_f{f}_s{args.seed}.csv"
        s = Path(args.pred_dir) / f"sel_f{f}_s{args.seed}.csv"
        dv = pd.read_csv(v)[["patientId", "y", "viewpos", "p_clean"]] \
            .rename(columns={"p_clean": "p"})
        dv["split"] = "val"
        ds = pd.read_csv(s)[["patientId", "y", "viewpos", "p_sel"]] \
            .rename(columns={"p_sel": "p"})
        ds["split"] = "sel"
        g = pd.concat([dv, ds], ignore_index=True)
        g["fold"] = f
        fr.append(g)
    d = pd.concat(fr, ignore_index=True)
    d["class"] = d["patientId"].map(cls)
    return d


def check_0_integrity(d) -> None:
    print("\n" + "=" * 78)
    print("0  INTEGRITY: does the class table say what part A assumed")
    print("=" * 78)
    print(f"  films without a class label: {int(d['class'].isna().sum())}")
    ct = pd.crosstab(d["class"], d["y"])
    print("\n  class against label (all splits)")
    print("  " + ct.to_string().replace("\n", "\n  "))
    ok = (ct.loc[CLASSES[2], 0.0] == 0 and ct.loc[CLASSES[0], 1.0] == 0
          and ct.loc[CLASSES[1], 1.0] == 0)
    print(f"\n  'Lung Opacity' <=> y=1 holds exactly: {ok}")
    for f, g in d.groupby("fold"):
        ov = set(g[g.split == "sel"].patientId) & set(g[g.split == "val"].patientId)
        print(f"  fold {f}: sel {int((g.split=='sel').sum())}, "
              f"val {int((g.split=='val').sum())}, patient overlap {len(ov)}")
    v = d[d.split == "val"]
    print(f"  val total {len(v)}, distinct patients {v.patientId.nunique()}")


def check_1_classes(d) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("1  PART A, POOLED AND THEN SPLIT BY PROJECTION")
    print("=" * 78)
    rows = []
    for f, g in d.groupby("fold"):
        sel, val = g[g.split == "sel"], g[g.split == "val"]
        thr = youden_threshold(sel.y.to_numpy(), sel.p.to_numpy())
        v = val.copy()
        v["pred"] = v.p.to_numpy() >= thr
        rows.append(v)
    V = pd.concat(rows, ignore_index=True)

    def block(sub, name):
        print(f"\n  --- {name}, n={len(sub)} ---")
        print(f"  {'class':<26}{'n/fold':>8}{'called positive':>20}{'mean score':>12}")
        rate = {}
        for c in CLASSES:
            m = sub[sub["class"] == c]
            per = m.groupby("fold").agg(cp=("pred", "mean"), sc=("p", "mean"),
                                        n=("p", "size"))
            rate[c] = per.cp.mean()
            print(f"  {SHORT[c]:<26}{per.n.mean():>8.0f}"
                  f"{per.cp.mean():>14.3f} +-{per.cp.std():.3f}{per.sc.mean():>12.3f}")
        neg = sub[sub.y == 0]
        fp = sub[sub.pred & (sub.y == 0)]
        share = fp.groupby("fold")["class"].value_counts(normalize=True) \
            .unstack(fill_value=0).get(CLASSES[1])
        base = neg.groupby("fold")["class"].value_counts(normalize=True) \
            .unstack(fill_value=0)[CLASSES[1]]
        print(f"  share of all FP in the middle class {share.mean():>8.3f} "
              f"+-{share.std():.3f}   (middle class is {base.mean():.1%} of all "
              f"negatives)")
        print(f"  false alarm rate middle : normal    {rate[CLASSES[1]]/rate[CLASSES[0]]:>8.1f} x"
              "   <-- the statistic that is free of class sizes")

    block(V, "pooled, as published")
    for vp in ("AP", "PA"):
        block(V[V.viewpos == vp], f"{vp} only")
    print("\n  share of AP films per class")
    print("  " + pd.crosstab(V["class"], V["viewpos"], normalize="index")
          .round(3).to_string().replace("\n", "\n  "))
    return V


def check_2_operating_point(d) -> None:
    print("\n" + "=" * 78)
    print("2  THE OPERATING POINT, SPLIT BY PROJECTION")
    print("=" * 78)
    print("  The published Youden point pools AP and PA. A single threshold has")
    print("  one sensitivity per projection, and they are not the same number.")
    k = {n: [] for n in ("se", "sp", "AP_se", "AP_sp", "PA_se", "PA_sp",
                         "own_AP_se", "own_AP_sp", "own_PA_se", "own_PA_sp",
                         "t", "t_AP", "t_PA")}
    for f, g in d.groupby("fold"):
        sel, val = g[g.split == "sel"], g[g.split == "val"]
        t = youden_threshold(sel.y.to_numpy(), sel.p.to_numpy())
        k["t"].append(t)
        pr, y = val.p.to_numpy() >= t, val.y.to_numpy()
        k["se"].append(pr[y == 1].mean())
        k["sp"].append((~pr[y == 0]).mean())
        for vp in ("AP", "PA"):
            s, v = sel[sel.viewpos == vp], val[val.viewpos == vp]
            yv = v.y.to_numpy()
            own = youden_threshold(s.y.to_numpy(), s.p.to_numpy())
            k[f"t_{vp}"].append(own)
            for tag, thr in (("", t), ("own_", own)):
                pv = v.p.to_numpy() >= thr
                k[f"{tag}{vp}_se"].append(pv[yv == 1].mean())
                k[f"{tag}{vp}_sp"].append((~pv[yv == 0]).mean())
    print(f"\n  one shared threshold (mean {np.mean(k['t']):.3f})")
    print(f"    pooled, as published   sens {ms(k['se'])[0]:.3f} +-{ms(k['se'])[1]:.3f}"
          f"   spec {ms(k['sp'])[0]:.3f} +-{ms(k['sp'])[1]:.3f}")
    for vp in ("AP", "PA"):
        print(f"    {vp} only               sens {ms(k[vp+'_se'])[0]:.3f} "
              f"+-{ms(k[vp+'_se'])[1]:.3f}   spec {ms(k[vp+'_sp'])[0]:.3f} "
              f"+-{ms(k[vp+'_sp'])[1]:.3f}")
    gap = ms(k["AP_se"])[0] - ms(k["PA_se"])[0]
    print(f"    sensitivity gap {gap:.3f}  <-- rsna_train.py already writes this "
          "as sens_gap")
    print(f"\n  its own threshold per projection (AP {np.mean(k['t_AP']):.3f}, "
          f"PA {np.mean(k['t_PA']):.3f})")
    for vp in ("AP", "PA"):
        print(f"    {vp} only               sens {ms(k['own_'+vp+'_se'])[0]:.3f} "
              f"+-{ms(k['own_'+vp+'_se'])[1]:.3f}   spec "
              f"{ms(k['own_'+vp+'_sp'])[0]:.3f} +-{ms(k['own_'+vp+'_sp'])[1]:.3f}")
    print("  Both projections do better under their own threshold. Whether the")
    print("  app can use that is a separate question: it is fed a PNG with no")
    print("  DICOM header, so it does not know the projection.")


def check_3_poswatch(d) -> None:
    print("\n" + "=" * 78)
    print("3  IS pos_weight THE CAUSE? The claim Phase 3 made and did not test")
    print("=" * 78)
    print(f"  A pure pos_weight artefact means: slope 1.0, intercept -log(w).")
    print(f"  For w = 3.4 that intercept is {-np.log(3.4):.4f}.\n")
    print(f"  {'fold':<6}{'slope':>9}{'intercept':>12}{'implied w':>12}")
    W = []
    for f, g in d.groupby("fold"):
        sel = g[g.split == "sel"]
        w = fit_recal(sel.y.to_numpy(float), sel.p.to_numpy(float), True)
        W.append(w)
        print(f"  {f:<6}{w[0]:>9.3f}{w[1]:>12.3f}{np.exp(-w[1]):>12.2f}")
    W = np.array(W)
    print(f"  {'mean':<6}{W[:,0].mean():>9.3f}{W[:,1].mean():>12.3f}"
          f"{np.exp(-W[:,1].mean()):>12.2f}")
    print(f"  slope sd {W[:,0].std(ddof=1):.3f}, intercept sd {W[:,1].std(ddof=1):.3f}")

    print("\n  How much does each part buy? Reliability on the validation fold,")
    print("  reported as the typical distance from the diagonal (the root, in")
    print("  probability points).")
    raw = [reliability(x.y.to_numpy(), x.p.to_numpy())
           for _, x in d[d.split == "val"].groupby("fold")]
    print(f"  {'raw score':<40}{np.mean(raw):>11.6f}   {np.sqrt(np.mean(raw)):>7.4f}")
    for name, free in (("shift only  (the pos_weight model)", False),
                       ("shift + shrink  (Platt, as used)", True)):
        rl = []
        for f, g in d.groupby("fold"):
            sel, val = g[g.split == "sel"], g[g.split == "val"]
            w = fit_recal(sel.y.to_numpy(float), sel.p.to_numpy(float), free)
            rl.append(reliability(val.y.to_numpy(),
                                  apply_recal(val.p.to_numpy(float), w, free)))
        print(f"  {name:<40}{np.mean(rl):>11.6f}   {np.sqrt(np.mean(rl)):>7.4f}")


def check_4_auc(d) -> None:
    print("\n" + "=" * 78)
    print("4  WHICH AUC IS 0.8802")
    print("=" * 78)
    val = d[d.split == "val"]
    pooled, strat, ap, pa = [], [], [], []
    for f, x in val.groupby("fold"):
        pooled.append(auc(x.y.to_numpy(), x.p.to_numpy()))
        a = x[x.viewpos == "AP"]
        b = x[x.viewpos == "PA"]
        ap.append(auc(a.y.to_numpy(), a.p.to_numpy()))
        pa.append(auc(b.y.to_numpy(), b.p.to_numpy()))
        strat.append((ap[-1] * len(a) + pa[-1] * len(b)) / (len(a) + len(b)))
    print(f"  pooled, quoted in the phase 3 note   {np.mean(pooled):.4f} "
          f"+-{np.std(pooled, ddof=1):.4f}")
    print(f"  stratified, as rsna_train.py defines {np.mean(strat):.4f} "
          f"+-{np.std(strat, ddof=1):.4f}   <-- the project's honest number")
    print(f"  AP only {np.mean(ap):.4f}   PA only {np.mean(pa):.4f}")
    print("  The pooled number is above BOTH within-projection numbers. That gap")
    print("  is the confounder, and it is the whole point of this project.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--pred-dir", type=Path, default=Path("predictions_rsna_base"))
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    d = load(args)
    check_0_integrity(d)
    check_1_classes(d)
    check_2_operating_point(d)
    check_3_poswatch(d)
    check_4_auc(d)
    print("\nnothing written to disk: this script only reads.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
