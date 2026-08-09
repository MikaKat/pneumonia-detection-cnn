"""Phase 5, the second head: does pointing cost anything, and which variant.

WHAT THIS PRODUCES
------------------
The pre-registered evaluation of phase 5, in the pre-registered order, plus the
provenance checks that have to pass before any number is allowed to mean
something. Nothing here trains. Input is the three prediction directories, the
location prior from phase 1, the lung masks and the box list.

  karten    the expensive half: per image, per arm, the localisation measures
            of the head field and of the location prior. Cached in a CSV, so it
            runs once and the report below is instant afterwards.
  bericht   the statistics: B, then A, then the map table, then the exploratory
            part. Reads the cache, writes the summary CSVs, prints the report.

WHY IT EXISTS
-------------
Phase 5 added a second output to the network, a 14 by 14 field that is trained
on the annotated boxes. The classification branch is unchanged. Two questions
follow, and they are not the same question:

  A  what does the head COST the classifier. Primary endpoint, stratified AUC,
     the n-weighted mean of the AP-only and the PA-only AUC (AUC is the
     c-statistic: the chance that a random film with pneumonia gets a higher
     score than a random one without).
  B  which of the two variants for images WITHOUT a box is better, the one that
     drops them from the head loss (exclude) or the one that asks the head for
     an empty field (empty).

B is not an endpoint. A model trained to point will point better than one that
was never asked to; that is a definition, not a finding. B decides one thing
only, namely which arm enters A.

HOW TO READ THE RESULT
----------------------
Endpoint A is a NON-INFERIORITY question, and the direction matters. The
pre-registration reads "if A falls by less than 0.01 the head is kept". Only the
LOWER end of the interval can decide that. An upper end above +0.01 means the
head is better than the margin, which is not a failure of a non-inferiority
test; reporting it as one would be reading an equivalence test that nobody
pre-registered. Both readings are printed, with the pre-registered one marked.

The null line for B is the LOCATION PRIOR, not chance. The prior is the averaged
box map of the fitting part: a map that knows nothing except where opacities
usually sit. Point AUC has a chance value of exactly 0.5 whatever the box area
is, but 0.5 belongs to a pointer that points anywhere, including at the air
beside the patient. Beating the prior is the question that matters.

"Not established" never means "no difference". It means the five folds cannot
resolve it. The half-width of every interval is printed for that reason.

CLI, from the repository root:
  python rsna\\befunde\\rsna_phase5_auswertung.py karten --folds 0 1
  python rsna\\befunde\\rsna_phase5_auswertung.py karten --folds 2 3 4
  python rsna\\befunde\\rsna_phase5_auswertung.py bericht
"""

from __future__ import annotations

import _repo_path  # noqa: F401

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_lokalisation import REF_SIZE, box_mask, load_boxes, load_lung, to_reference

# ---- the pre-registration, in code so it cannot drift from the prose --------
DELTA_A = 0.01        # non-inferiority margin on the stratified AUC
DELTA_C = 0.02        # endpoint C is a diagnostic number, twice the margin
ALPHA = 0.10          # 90 percent interval, the usual choice for TOST at 5 %
BOOT = 2000           # bootstrap draws for the per-image intervals
SEED = 0
MIN_N = 30            # below this no verdict is printed

ARMS = {
    "ref": "predictions_p5_ref",
    "ex":  "predictions_final_model",
    "em":  "predictions_p5_head_em",
}
HEAD_ARMS = ("ex", "em")
TAGS = {"_p5ref": "ref", "_p5head_ex": "ex", "_p5head_em": "em"}

FINDINGS: list[str] = []


def check(name: str, cond: bool, info: str = "") -> bool:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FINDINGS.append(name)
    return cond


def note(text: str) -> None:
    print(f"        {text}")


# --------------------------------------------------------------------------
# Statistics. The two rank measures are written out here rather than imported
# so that a disagreement with the training script is a disagreement and not a
# shared bug. The t quantile is the only place scipy would be needed, and it is
# reproduced below so this script runs anywhere the prediction files run.
# --------------------------------------------------------------------------

def rank_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC by mid ranks. Same value as sklearn.metrics.roc_auc_score."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    new = np.empty(ps.size, dtype=bool)
    new[0] = True
    np.not_equal(ps[1:], ps[:-1], out=new[1:])
    grp = np.cumsum(new) - 1
    counts = np.bincount(grp)
    ends = np.cumsum(counts)
    starts = ends - counts
    mid = 0.5 * (starts + 1 + ends)
    ranks = np.empty(ps.size, dtype=np.float64)
    ranks[order] = mid[grp]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def point_auc(heat: np.ndarray, box: np.ndarray,
              valid: np.ndarray | None = None) -> float:
    """Pixel level AUC of a map against the boxes, mid ranks for ties.

    Positive class are the pixels inside a box, negative class the rest, and
    `valid` restricts both to the lung. Mid ranks are not cosmetic here: a map
    upsampled from a coarse grid consists of plateaus, and ordering ties would
    invent information the map does not carry.
    """
    h = np.asarray(heat, dtype=np.float64).ravel()
    b = np.asarray(box, dtype=bool).ravel()
    if valid is not None:
        v = np.asarray(valid, dtype=bool).ravel()
        h, b = h[v], b[v]
    if b.all() or not b.any():
        return float("nan")
    return rank_auc(b.astype(int), h)


def stratified_auc(y: np.ndarray, p: np.ndarray, vp: np.ndarray) -> float:
    """n-weighted mean of the AP-only and the PA-only AUC.

    The overall AUC still contains the projection effect, which is the whole
    confounder of this project. Only this number says something about
    radiology.
    """
    parts, w = [], []
    for v in ("AP", "PA"):
        m = vp == v
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        parts.append(rank_auc(y[m], p[m]))
        w.append(int(m.sum()))
    if len(parts) < 2:
        return float("nan")
    w = np.asarray(w, float)
    return float(np.dot(parts, w) / w.sum())


def score_to_view(p: np.ndarray, vp: np.ndarray) -> float:
    """Endpoint C, AUC(score -> ViewPosition). How much projection is left in
    the score. Not folded to max(a, 1 - a): the direction carries meaning."""
    m = np.isin(vp, ["AP", "PA"])
    if not m.sum() or len(set(vp[m])) < 2:
        return float("nan")
    return rank_auc((vp[m] == "AP").astype(int), p[m])


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function, Lentz's method."""
    tiny, eps, itmax = 1e-300, 3e-16, 300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_ppf(p: float, df: int) -> float:
    """Quantile of Student's t. Agrees with scipy.stats.t.ppf to 1e-12.

    Written out because it is the only reason this script would need scipy, and
    a report that cannot be recomputed on a plain interpreter is a report that
    stops being recomputed. `bericht --selbsttest` checks it against scipy
    whenever scipy happens to be installed.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    lo, hi = 0.0, 1.0
    target = 2.0 * (1.0 - p)
    while _betai(df / 2.0, 0.5, df / (df + hi * hi)) > target:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _betai(df / 2.0, 0.5, df / (df + mid * mid)) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paired_t(d: np.ndarray, alpha: float = ALPHA) -> dict:
    """Mean paired difference with a two-sided (1 - alpha) interval."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return {"n": n, "mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "sd": float("nan"), "half": float("nan")}
    m = float(d.mean())
    sd = float(d.std(ddof=1))
    half = t_ppf(1.0 - alpha / 2.0, n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "lo": m - half, "hi": m + half,
            "half": half}


def paired_boot(d: np.ndarray, seed: int = SEED, boot: int = BOOT) -> dict:
    """Mean paired difference with a 95 percent bootstrap interval.

    Bootstrap and not a t interval because the per-image point AUC is skewed:
    it is bounded above by 1 and many images sit close to it.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return {"n": n, "mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "sd": float("nan")}
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, n, size=(boot, n))].mean(axis=1)
    return {"n": n, "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
            "lo": float(np.percentile(m, 2.5)),
            "hi": float(np.percentile(m, 97.5))}


def verdict_noninferior(r: dict, delta: float, label: str) -> str:
    """The pre-registered reading, and the equivalence reading beside it.

    Non-inferiority looks at the lower end only. Equivalence looks at both. The
    two disagree exactly when the new arm is BETTER than the margin, and in that
    case reporting the equivalence verdict alone would turn a gain into a
    failure.
    """
    lo, hi, m = r["lo"], r["hi"], r["mean"]
    ni = "PASS" if lo > -delta else ("FAIL" if hi <= -delta else "UNCLEAR")
    if -delta < lo and hi < delta:
        eq = "PASS"
    elif lo >= delta or hi <= -delta:
        eq = "FAIL"
    else:
        eq = "UNCLEAR"
    print(f"  {label}: mean difference {m:+.4f}, sd {r['sd']:.4f}, "
          f"{int((1 - ALPHA) * 100)} % interval [{lo:+.4f}, {hi:+.4f}], "
          f"margin {delta:.3f}")
    print(f"  {label}: NON-INFERIORITY {ni}   (equivalence would read {eq})")
    if ni == "PASS" and lo > 0:
        note("The interval sits entirely above zero, so the head did not cost")
        note("anything on this endpoint, it gained. Superiority was NOT")
        note("pre-registered, so this is a lead to confirm, not a result.")
    if ni == "PASS" and eq != "PASS":
        note("The equivalence reading is UNCLEAR only because the upper end")
        note("exceeds the margin, which is the good direction. The")
        note("pre-registration asks one question: does A fall by more than the")
        note("margin. It does not.")
    if ni == "UNCLEAR":
        note("Not a pass and not a refutation. The five folds cannot resolve")
        note(f"it. Half-width {r['half']:.4f}: nothing smaller is measurable")
        note("with this many folds.")
        FINDINGS.append(f"{label}: too imprecise for a verdict")
    if ni == "FAIL":
        FINDINGS.append(f"{label}: the endpoint fell by more than the margin")
    return ni


def verdict_c(r: dict, delta: float, label: str) -> str:
    """Endpoint C, where the good direction is DOWN, not up.

    C is the share of projection information left in the score, and the whole
    project is trying to push it down. The non-inferiority reading used for A
    would therefore be upside down here: on A the worry is a fall, on C it is a
    RISE. Reading A's verdict function on C would call an interval that reaches
    +0.023 a pass, which is exactly the direction that would worry anyone.

    Reported are the equivalence verdict (did the head move C at all) and the
    one-sided question that matters (did C rise by more than the margin).
    """
    lo, hi, m = r["lo"], r["hi"], r["mean"]
    if -delta < lo and hi < delta:
        eq = "PASS"
    elif lo >= delta or hi <= -delta:
        eq = "FAIL"
    else:
        eq = "UNCLEAR"
    kein_anstieg = hi < delta
    print(f"  {label}: mean difference {m:+.4f}, sd {r['sd']:.4f}, "
          f"{int((1 - ALPHA) * 100)} % interval [{lo:+.4f}, {hi:+.4f}], "
          f"margin {delta:.3f}")
    print(f"  {label}: EQUIVALENCE {eq}, "
          f"'C did not rise beyond the margin' "
          f"{'holds' if kein_anstieg else 'NOT established'}")
    if not kein_anstieg:
        note("The upper end of the interval sits above the margin, so a rise")
        note("of C larger than the project would act on cannot be ruled out.")
        note("C is exploratory here, so this decides nothing, but it is the")
        note("direction that would matter and it is not clean.")
        FINDINGS.append(f"{label}: a rise beyond the margin cannot be excluded")
    if hi < 0:
        note("The whole interval sits below zero: the head pushed C down,")
        note("which is the direction the project wants.")
    return eq


# --------------------------------------------------------------------------
# The expensive half
# --------------------------------------------------------------------------

def karten_fold(arm: str, fold: int, args, boxes: dict, sp: dict) -> pd.DataFrame:
    """Head field and location prior on every annotated validation image."""
    pred = Path(ARMS[arm])
    f = pred / f"head_f{fold}_s{args.seed}.npz"
    prior_p = Path(args.baselines) / f"prior_f{fold}.npy"
    if not f.exists():
        print(f"  {arm} fold {fold}: {f} missing, skipped")
        return pd.DataFrame()
    if not prior_p.exists():
        raise SystemExit(f"ABORT: {prior_p} missing. It comes from phase 1:\n"
                         f"  python rsna\\befunde\\rsna_lokalisation.py tor")

    val_ids = sp["folds"][fold]["val"]
    z = np.load(f, allow_pickle=False)
    got = [str(x) for x in z["patientId"]]
    if got != list(val_ids):
        raise SystemExit(
            f"ABORT: {f} holds {len(got)} ids, the fold has {len(val_ids)}, and "
            f"they do not match. This file belongs to another fold or split.")
    fields = z["field"]
    prior = np.load(prior_p)
    vpmap = sp["viewpos"]

    rows, no_mask = [], 0
    pos = [(k, i) for k, i in enumerate(val_ids) if i in boxes]
    for k, pid in pos:
        lung = load_lung(args.masks, pid)
        if lung is None:
            no_mask += 1
            continue
        b = box_mask(boxes[pid], REF_SIZE)
        head = to_reference(fields[k], REF_SIZE)
        for name, heat in (("Kopf", head), ("Lagepriore", prior)):
            h = np.clip(np.asarray(heat, dtype=np.float64), 0, None)
            total = float(h.sum())
            deg = bool(total <= 0)
            yx = np.unravel_index(int(np.argmax(h)), h.shape)
            rows.append({
                "arm": arm, "fold": fold, "patientId": pid, "map": name,
                "viewpos": vpmap.get(pid, "?"),
                "point_auc": point_auc(h, b),
                "point_auc_lung": point_auc(h, b, lung),
                "hit": (not deg) and bool(b[yx]),
                "mass": 0.0 if deg else float(h[b].sum() / total),
                "area": float(b.mean()),
                "lung_area": float(np.asarray(lung, bool).mean()),
                "degenerate": deg,
            })
    print(f"  {arm} fold {fold}: {len(pos)} annotated images, "
          f"{no_mask} without a lung mask")
    return pd.DataFrame(rows)


def run_karten(args) -> int:
    sp = json.loads(Path(args.splits).read_text())
    boxes = load_boxes(args.csv)
    out = Path(args.cache)
    out.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()

    todo = []
    for arm in args.arms:
        for fold in args.folds:
            drin = (not old.empty
                    and ((old["arm"] == arm) & (old["fold"] == fold)).any())
            if drin and not args.force:
                print(f"  {arm} fold {fold}: already in the cache, skipped")
                continue
            todo.append((arm, fold))

    # Erst alle betroffenen Zeilen aus dem Zwischenspeicher nehmen, dann
    # rechnen. Wer das je Arm innerhalb der Schleife tut, wirft beim zweiten Arm
    # die gerade berechneten Zeilen des ersten wieder mit weg.
    if todo and not old.empty:
        weg = pd.Series(False, index=old.index)
        for arm, fold in todo:
            weg |= (old["arm"] == arm) & (old["fold"] == fold)
        old = old[~weg]

    parts = ([old] if not old.empty else [])
    parts += [karten_fold(arm, fold, args, boxes, sp) for arm, fold in todo]
    parts = [p for p in parts if not p.empty]
    if not parts:
        print("nothing computed and nothing in the cache")
        return 2
    d = pd.concat(parts, ignore_index=True)
    d = d.drop_duplicates(subset=["arm", "fold", "patientId", "map"], keep="last")
    d.to_csv(out, index=False)
    print(f"\n{len(d)} rows in {out}")
    have = d.groupby("arm")["fold"].unique().to_dict()
    for a, f in have.items():
        print(f"  {a}: folds {sorted(int(x) for x in f)}")
    return 0


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------

def load_preds(arm: str, fold: int, seed: int) -> pd.DataFrame:
    f = Path(ARMS[arm]) / f"rsna_f{fold}_s{seed}.csv"
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f).set_index("patientId").sort_index()


def step0_provenance(args) -> pd.DataFrame:
    """File names have lied in this project before. The rows have to say it."""
    print("\n0  WHERE DO THE ROWS SAY THEY COME FROM")
    r = pd.read_csv(args.results)
    if "tag" not in r.columns:
        check("results_rsna.csv carries tag / pred_dir / ckpt", False)
        return pd.DataFrame()
    r = r[r["tag"].isin(TAGS)].copy()
    r["arm"] = r["tag"].map(TAGS)
    check("all three arms are present",
          sorted(r["arm"].unique()) == ["em", "ex", "ref"],
          f"{sorted(r['arm'].unique())}")
    for arm in ARMS:
        sub = r[r["arm"] == arm]
        check(f"{arm}: five folds",
              sorted(set(int(x) for x in sub["fold"])) == sorted(args.folds),
              f"folds {sorted(set(int(x) for x in sub['fold']))}, "
              f"{len(sub)} row(s)")
    # A fold may legitimately appear twice: the run is restartable, and a fold
    # that had to be repeated leaves its old row behind. What counts is the
    # LAST row, because the prediction files it points at were overwritten by
    # that run. Silently keeping both would average a discarded run into the
    # result, so the repeats are named here instead of dropped quietly.
    dup = r[r.duplicated(subset=["arm", "fold"], keep=False)]
    if not dup.empty:
        print("\n  NOTE  a fold was run more than once:")
        for a_, f_ in sorted({(x, int(y)) for x, y
                              in dup[["arm", "fold"]].itertuples(index=False)}):
            rows = r[(r["arm"] == a_) & (r["fold"] == f_)]
            print(f"    {a_} fold {f_}: {len(rows)} rows, lambda "
                  + ", ".join(f"{v:.4g}" for v in rows["head_lambda"]))
            note("The LAST of them is used, because it wrote the prediction")
            note("files. The earlier one stays in the CSV as a record.")
    r = r.drop_duplicates(subset=["arm", "fold"], keep="last").copy()
    check("every row ran on adapter 1", bool((r["dml_index"] == 1).all()),
          f"dml_index {sorted(set(r['dml_index'].astype(int)))}")
    check("every row names the card",
          all("RX 5500" in str(x) for x in r["device_name"]))
    # Tolerance and not equality: the residual AUC is computed from weights and
    # comes back as 0.5000000000000001 in some rows. Testing it with == turns a
    # float artefact into a finding, which is a defect of the test.
    resid = np.abs(r["balance_residual_auc"].to_numpy() - 0.5).max()
    check("every row decoupled projection and label",
          bool((r["balance_view"] == 1).all()) and resid < 1e-9,
          f"balance_residual_auc deviates from 0.500 by at most {resid:.1e}")
    check("every row writes into its own directory",
          r.groupby("arm")["pred_dir"].nunique().eq(1).all()
          and r["pred_dir"].nunique() == 3)
    ck = r["ckpt"].astype(str)
    check("no two rows share a checkpoint file", ck.nunique() == len(ck))
    head = r[r["arm"].isin(HEAD_ARMS)]
    check("both head arms trained a 14 by 14 head",
          bool((head["head"] == 1).all() and (head["head_grid"] == 14).all()))
    check("the reference arm trained no head",
          bool((r[r["arm"] == "ref"]["head"] == 0).all()))

    # lambda is the one number that can silently turn one fold into a different
    # experiment, so it gets its own check rather than a glance at the log.
    lam = head[["arm", "fold", "head_lambda"]].copy()
    bad = lam[(lam["head_lambda"] < 0.1) | (lam["head_lambda"] > 10.0)]
    check("lambda stayed within a plausible range in every fold", bad.empty,
          "" if bad.empty else
          "; ".join(f"{a} fold {int(f)}: {v:.4g}"
                    for a, f, v in bad.itertuples(index=False)))
    if not bad.empty:
        note("lambda is measured on the FIRST batch as classification loss over")
        note("localisation loss, then frozen. With --head-negatives exclude an")
        note("image without a box contributes nothing, so a first batch that")
        note("happens to hold no annotated image gives a localisation loss of")
        note("exactly zero. The clamp at 1e-8 then turns the ratio into about")
        note("1e8, and that fold trains its trunk almost purely on the")
        note("localisation task. It is a protocol deviation, not a crash, and")
        note("nothing in the run reports it.")
        # The guard in rsna_train.loc_loss carries a prose estimate of how
        # often this happens. Prose estimates are worth what they are checked
        # against, so it is recomputed here from the numbers of this run.
        q = (1.0 - args.pos_share) ** args.batch
        any5 = 1.0 - (1.0 - q) ** len(args.folds)
        note("")
        note(f"How rare is it really: with {args.pos_share:.3f} annotated "
             f"images and batch {args.batch},")
        note(f"P(first batch without one) = {1 - args.pos_share:.3f}^"
             f"{args.batch} = {q:.4f}, about one in {1 / q:.0f}. Over "
             f"{len(args.folds)} folds")
        note(f"the chance that at least one is hit is {any5:.3f}. The docstring "
             f"of loc_loss")
        note("puts it at one in 50,000, which is what made this look like a")
        note("guard rather than something that needs handling.")
    # Which batch lambda came from. Written by the repaired rsna_train.py;
    # older rows leave it empty, and empty is not the same as 1. A value above
    # 1 is not a defect, it is the repair doing its work, so this reports
    # rather than judges.
    if "head_lambda_batch" in head.columns:
        b = pd.to_numeric(head["head_lambda_batch"], errors="coerce")
        late = head[b.notna() & (b > 1)]
        if not late.empty:
            print("\n  lambda was measured after batch 1 in:")
            for a, f, v in late[["arm", "fold", "head_lambda_batch"]].itertuples(
                    index=False):
                print(f"    {a} fold {int(f)}: batch {int(v)}")
            note("The first batch carried no annotated image and the")
            note("measurement waited for one. Batches before it give the head")
            note("no gradient, so nothing else about the fold changes.")

    print("\n  lambda per fold:")
    for a, f, v in lam.itertuples(index=False):
        print(f"    {a} fold {int(f)}: {v:.4g}")
    suspect = [(a, int(f)) for a, f, _ in bad.itertuples(index=False)]
    return r, suspect


def step1_b(d: pd.DataFrame, args) -> str:
    """B: which head variant points better. Decides who enters A, nothing else."""
    print("\n" + "=" * 74)
    print("1  B: THE HEAD FIELD AGAINST THE LOCATION PRIOR")
    print("=" * 74)
    print("  Point AUC inside the lung mask, paired per image, 95 % bootstrap.")
    print("  B selects the arm for A. It is not an endpoint: a model trained to")
    print("  point points better, which is a definition.\n")

    w = d.pivot_table(index=["arm", "fold", "patientId", "viewpos"],
                      columns="map", values="point_auc_lung").dropna()
    w = w.reset_index()
    w["Vorsprung"] = w["Kopf"] - w["Lagepriore"]

    rows = []
    print(f"  {'arm':>4} {'group':>10} {'n':>6} {'head':>8} {'prior':>8}"
          f" {'lead':>9} {'95 % interval':>22}")
    for arm in HEAD_ARMS:
        s = w[w["arm"] == arm]
        for label, sub in ([("pooled", s)]
                           + [(f"only {v}", s[s["viewpos"] == v])
                              for v in ("AP", "PA")]
                           + [(f"fold {f}", s[s["fold"] == f])
                              for f in sorted(s["fold"].unique())]):
            if len(sub) < MIN_N:
                continue
            r = paired_boot(sub["Vorsprung"].to_numpy())
            rows.append({"arm": arm, "gruppe": label, "n": r["n"],
                         "kopf": float(sub["Kopf"].mean()),
                         "priore": float(sub["Lagepriore"].mean()),
                         "vorsprung": r["mean"], "lo": r["lo"], "hi": r["hi"]})
            print(f"  {arm:>4} {label:>10} {r['n']:>6} "
                  f"{sub['Kopf'].mean():>8.4f} {sub['Lagepriore'].mean():>8.4f} "
                  f"{r['mean']:>+9.4f} [{r['lo']:>+8.4f}, {r['hi']:>+8.4f}]")

    gate = pd.DataFrame(rows)
    fails = gate[(gate["gruppe"].str.startswith("only")
                  | (gate["gruppe"] == "pooled")) & (gate["lo"] <= 0)]
    print()
    check("both arms beat the prior, pooled and in each projection",
          fails.empty,
          "" if fails.empty else
          "; ".join(f"{a} {g}" for a, g in
                    fails[["arm", "gruppe"]].itertuples(index=False)))

    # ---- the actual selection, paired image by image ----------------------
    print("\n  exclude against empty, paired per image")
    p = w.pivot_table(index=["fold", "patientId", "viewpos"], columns="arm",
                      values="Vorsprung").dropna()
    r_img = paired_boot((p["ex"] - p["em"]).to_numpy())
    print(f"    per image: n {r_img['n']}, difference in the lead "
          f"{r_img['mean']:+.4f} [{r_img['lo']:+.4f}, {r_img['hi']:+.4f}]")
    # The image level interval treats images as independent, which they are not:
    # five models produced them. The fold level test uses the unit that was
    # actually randomised and is the conservative one.
    per_fold = w.pivot_table(index="fold", columns="arm", values="Vorsprung")
    r_fold = paired_t((per_fold["ex"] - per_fold["em"]).to_numpy())
    print(f"    per fold:  n {r_fold['n']}, difference {r_fold['mean']:+.4f} "
          f"[{r_fold['lo']:+.4f}, {r_fold['hi']:+.4f}] (90 %)")
    print(f"    lead per fold: "
          + ", ".join(f"f{int(f)} ex {row['ex']:+.4f} em {row['em']:+.4f}"
                      for f, row in per_fold.iterrows()))

    winner = "ex" if r_img["mean"] > 0 else "em"
    loser = "em" if winner == "ex" else "ex"
    print(f"\n  WINNER OF B: {winner}")
    if not (r_img["lo"] > 0 or r_img["hi"] < 0):
        note("The interval covers zero. The point estimate still decides,")
        note("because the pre-registration asks for a selection and not for a")
        note("significant one, but the two variants are not distinguishable")
        note("here and the choice should not be sold as a finding.")
        FINDINGS.append("B: the two head variants are not separated by their lead")
    gate.to_csv(Path(args.out_dir) / "phase5_B_kopf_gegen_priore.csv", index=False)
    return winner, loser


def step2_a(args, winner: str, loser: str) -> pd.DataFrame:
    """A: the primary endpoint. Only the winner of B is confirmatory."""
    print("\n" + "=" * 74)
    print("2  A: WHAT THE HEAD COSTS THE CLASSIFIER")
    print("=" * 74)
    rows = []
    for arm in ARMS:
        for fold in args.folds:
            t = load_preds(arm, fold, args.seed)
            if t.empty:
                FINDINGS.append(f"{arm} fold {fold}: no prediction file")
                continue
            y = t["y"].to_numpy()
            vp = t["viewpos"].to_numpy().astype(str)
            p = t["p_clean"].to_numpy()
            rows.append({"arm": arm, "fold": fold, "n": len(t),
                         "A": stratified_auc(y, p, vp),
                         "A_AP": rank_auc(y[vp == "AP"], p[vp == "AP"]),
                         "A_PA": rank_auc(y[vp == "PA"], p[vp == "PA"]),
                         "C": score_to_view(p, vp),
                         "auc_raw": rank_auc(y, p)})
    d = pd.DataFrame(rows)
    # Every arm has to hold every fold. A missing one would not crash: the
    # paired test drops non-finite differences and would quietly compare four
    # folds of one arm with five of the other, which is no longer a cross-over
    # and no longer answers the question. This is the state the repository is
    # in between archiving a fold and recomputing it.
    fehlt = sorted({(arm, f) for arm in ARMS for f in args.folds}
                   - {(t.arm, t.fold) for t in d.itertuples()})
    if fehlt:
        raise SystemExit(
            "ABORT: prediction files missing for "
            + ", ".join(f"{a} fold {f}" for a, f in fehlt)
            + ".\nA paired comparison needs both arms on every fold. Run the "
              "missing fold first, then this report again.")
    A = d.pivot(index="fold", columns="arm", values="A")
    C = d.pivot(index="fold", columns="arm", values="C")

    print(f"\n  {'fold':>4} {'ref':>8} {'ex':>8} {'em':>8} "
          f"{'ex - ref':>9} {'em - ref':>9}")
    for f in A.index:
        print(f"  {int(f):>4} {A.loc[f,'ref']:>8.4f} {A.loc[f,'ex']:>8.4f} "
              f"{A.loc[f,'em']:>8.4f} {A.loc[f,'ex']-A.loc[f,'ref']:>+9.4f} "
              f"{A.loc[f,'em']-A.loc[f,'ref']:>+9.4f}")
    print(f"  {'mean':>4} {A['ref'].mean():>8.4f} {A['ex'].mean():>8.4f} "
          f"{A['em'].mean():>8.4f} {(A['ex']-A['ref']).mean():>+9.4f} "
          f"{(A['em']-A['ref']).mean():>+9.4f}")

    # The same table split by projection. A number that survives pooled and
    # falls apart per stratum was never a finding.
    print("\n  the same, per projection")
    for col, name in (("A_AP", "AP only"), ("A_PA", "PA only")):
        P = d.pivot(index="fold", columns="arm", values=col)
        print(f"    {name:>8}: ref {P['ref'].mean():.4f}  "
              f"ex {P['ex'].mean():.4f} ({(P['ex']-P['ref']).mean():+.4f})  "
              f"em {P['em'].mean():.4f} ({(P['em']-P['ref']).mean():+.4f})")

    print(f"\n  CONFIRMATORY, the winner of B against the reference arm:")
    verdict_noninferior(paired_t((A[winner] - A["ref"]).to_numpy()), DELTA_A,
                        f"A, {winner} - ref")
    print(f"\n  EXPLORATORY, the loser of B:")
    verdict_noninferior(paired_t((A[loser] - A["ref"]).to_numpy()), DELTA_A,
                        f"A, {loser} - ref")

    print("\n  C, AUC(score -> ViewPosition), exploratory")
    print(f"  {'fold':>4} {'ref':>8} {'ex':>8} {'em':>8}")
    for f in C.index:
        print(f"  {int(f):>4} {C.loc[f,'ref']:>8.4f} {C.loc[f,'ex']:>8.4f} "
              f"{C.loc[f,'em']:>8.4f}")
    print(f"  {'mean':>4} {C['ref'].mean():>8.4f} {C['ex'].mean():>8.4f} "
          f"{C['em'].mean():>8.4f}")
    for arm in HEAD_ARMS:
        verdict_c(paired_t((C[arm] - C["ref"]).to_numpy()), DELTA_C,
                  f"C, {arm} - ref")
    d.to_csv(Path(args.out_dir) / "phase5_A_je_fold.csv", index=False)
    return d


def step3_karten(d: pd.DataFrame, args) -> None:
    """The three map table. Every row next to the location prior."""
    print("\n" + "=" * 74)
    print("3  THREE MAPS, EACH BESIDE THE LOCATION PRIOR")
    print("=" * 74)
    prior = d[d["map"] == "Lagepriore"].groupby("fold")["point_auc_lung"].mean()
    print(f"\n  location prior, point AUC in the lung: {prior.mean():.4f}")

    print("\n  a) head output, point AUC in the lung")
    for arm in HEAD_ARMS:
        s = d[(d["arm"] == arm) & (d["map"] == "Kopf")]
        print(f"     {arm}: {s['point_auc_lung'].mean():.4f}  "
              f"(AP {s[s.viewpos=='AP']['point_auc_lung'].mean():.4f}, "
              f"PA {s[s.viewpos=='PA']['point_auc_lung'].mean():.4f})")

    print("\n  b) Grad-CAM, hit rate and mass against the chance level")
    print(f"     {'arm':>4} {'n':>6} {'hit':>8} {'chance':>8} {'lift':>8}"
          f" {'mass':>8} {'lift':>8} {'degenerate':>11}")
    cam_rows = []
    for arm in ARMS:
        parts = []
        for f in args.folds:
            p = Path(ARMS[arm]) / f"cam_f{f}_s{args.seed}.csv"
            if p.exists():
                t = pd.read_csv(p)
                t["fold"] = f
                parts.append(t)
        if not parts:
            continue
        t = pd.concat(parts, ignore_index=True)
        row = {"arm": arm, "n": len(t), "hit": t["hit"].mean(),
               "area": t["area"].mean(), "mass": t["mass"].mean(),
               "degenerate": int(t["degenerate"].sum())}
        cam_rows.append(row)
        print(f"     {arm:>4} {row['n']:>6} {row['hit']:>8.4f} "
              f"{row['area']:>8.4f} {row['hit']-row['area']:>+8.4f} "
              f"{row['mass']:>8.4f} {row['mass']-row['area']:>+8.4f} "
              f"{row['degenerate']:>11}")
    cam = pd.DataFrame(cam_rows)
    cam.to_csv(Path(args.out_dir) / "phase5_gradcam_treffer.csv", index=False)
    # A degenerate map is one that is zero everywhere. It counts as a miss, so
    # an arm that produces more of them is being punished for something that is
    # not a localisation failure but a dead Grad-CAM.
    worst = cam.loc[cam["degenerate"].idxmax()]
    check("no arm produces markedly more degenerate Grad-CAM maps",
          float(worst["degenerate"]) <= 2 * float(cam["degenerate"].median()),
          f"{worst['arm']} has {int(worst['degenerate'])} of {int(worst['n'])}, "
          f"median over the arms {cam['degenerate'].median():.0f}")
    note("hit and mass are the numbers the training run already writes. They")
    note("are not point AUC and they are not comparable with row a), because")
    note("their chance level moves with the box area. The point AUC row for")
    note("Grad-CAM needs the maps themselves and therefore a rerun:")
    note("  python rsna\\befunde\\rsna_cam_power.py --arms p5ref p5head_ex "
         "p5head_em")
    note("Until then the middle row of the three map table is missing and the")
    note("comparison emergent against trained rests on hit and mass alone.")

    # phase 2 measured the single headed Grad-CAM with point AUC, on the same
    # images. Different run, so it is context and not a paired comparison.
    p2 = Path(args.phase2)
    if p2.exists():
        s = pd.read_csv(p2)
        print("\n  c) for context, phase 2 measured single headed Grad-CAM with")
        print("     point AUC on the same images (adapter 0, other run):")
        for arm in ("base", "bal10", "Lagepriore"):
            sub = s[s["arm"] == arm]
            if not sub.empty:
                print(f"     {arm:>12}: {sub['point_auc_lung'].mean():.4f}")
        note("Not paired with anything in phase 5. It stands here because it is")
        note("the only existing point AUC of a Grad-CAM map in this project,")
        note("and because it is the central phase 2 result: Grad-CAM sits BELOW")
        note("the location prior on average, the head sits far above it.")


def step4_kreuzprobe(d: pd.DataFrame, a: pd.DataFrame, args) -> None:
    """Every number in the report against the file it should come from."""
    print("\n" + "=" * 74)
    print("4  CROSS CHECK")
    print("=" * 74)
    r = pd.read_csv(args.results)
    r = r[r["tag"].isin(TAGS)].copy()
    r["arm"] = r["tag"].map(TAGS)
    # Same rule as in step 0: a repeated fold leaves two rows and the last one
    # is the one whose prediction files are on disk.
    r = r.drop_duplicates(subset=["arm", "fold"], keep="last")
    m = r.merge(a, on=["arm", "fold"], suffixes=("_run", "_hier"))
    for col_run, col_here, name in (("auc_stratified", "A", "A, stratified AUC"),
                                    ("auc_view", "C", "C, score -> view"),
                                    ("auc", "auc_raw", "raw AUC")):
        dev = float(np.abs(m[col_run] - m[col_here]).max())
        check(f"{name} recomputed from the prediction files", dev < 1e-9,
              f"largest deviation {dev:.2e}")

    off = Path(args.offiziell)
    if off.exists():
        o = pd.read_csv(off)
        j = d[d["map"].isin(("Kopf", "Lagepriore"))].merge(
            o, on=["arm", "fold", "patientId", "map"], suffixes=("_hier", "_off"))
        if j.empty:
            check("cross check against rsna_kopf_auswertung.py", False,
                  "no shared rows")
        else:
            dev = float(np.abs(j["point_auc_lung_hier"]
                               - j["point_auc_lung_off"]).max())
            check("point AUC identical to rsna_kopf_auswertung.py", dev == 0.0,
                  f"{len(j)} rows, largest deviation {dev:.2e}")
    else:
        note(f"{off} not present, the second implementation was not compared.")

    n = d.groupby(["arm", "map"]).size().unstack()
    check("both arms scored the same images",
          bool(n.nunique().eq(1).all()), f"{n.to_dict()}")


def step5_sensitivitaet(d: pd.DataFrame, a: pd.DataFrame, suspect: list,
                        winner: str, loser: str) -> None:
    """Does either conclusion depend on the fold that ran with a broken lambda?

    A deviation that is only described is not handled. The question a reader
    will ask is whether the result survives without that fold, so it gets
    answered here rather than left as a caveat.
    """
    if not suspect:
        return
    print("\n" + "=" * 74)
    print("5  SENSITIVITY: LEAVING OUT THE FOLD WITH THE BROKEN LAMBDA")
    print("=" * 74)
    for arm, fold in suspect:
        print(f"\n  dropping {arm} fold {fold}")
        keep = [f for f in sorted(a["fold"].unique()) if f != fold]

        w = d[d["fold"].isin(keep)].pivot_table(
            index=["arm", "fold", "patientId"], columns="map",
            values="point_auc_lung").dropna().reset_index()
        w["Vorsprung"] = w["Kopf"] - w["Lagepriore"]
        p = w.pivot_table(index=["fold", "patientId"], columns="arm",
                          values="Vorsprung").dropna()
        r = paired_boot((p["ex"] - p["em"]).to_numpy())
        lead = w.groupby("arm")["Vorsprung"].mean()
        print(f"    B: ex {lead['ex']:+.4f}, em {lead['em']:+.4f}, "
              f"difference {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
        still = "ex" if r["mean"] > 0 else "em"
        check(f"B still selects {winner} without that fold", still == winner,
              f"selects {still}")

        A = a[a["fold"].isin(keep)].pivot(index="fold", columns="arm", values="A")
        for arm2, tag in ((winner, "winner"), (loser, "loser")):
            rr = paired_t((A[arm2] - A["ref"]).to_numpy())
            ni = "PASS" if rr["lo"] > -DELTA_A else "not PASS"
            print(f"    A ({tag} {arm2}): {rr['mean']:+.4f} "
                  f"[{rr['lo']:+.4f}, {rr['hi']:+.4f}]  non-inferiority {ni}")
            if arm2 == winner:
                check("A still passes without that fold", ni == "PASS")

        Call = a.pivot(index="fold", columns="arm", values="C")
        C = Call.loc[keep]
        print(f"    C ({winner} - ref): {(C[winner] - C['ref']).mean():+.4f} "
              f"without that fold, "
              f"{(Call[winner] - Call['ref']).mean():+.4f} with all folds")
        note("C is the endpoint that changes sign here, which means the whole")
        note("apparent gain of that arm on C came from this one fold. The fold")
        note("is also the one whose trunk trained almost only on the boxes, so")
        note("the reading 'heavy localisation weight pushes the projection out")
        note("of the score' is a lead worth one deliberate run. One fold, and")
        note("the mechanism was an accident, so it is not a result.")


def run_bericht(args) -> int:
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.selbsttest:
        try:
            from scipy import stats
            worst = max(abs(t_ppf(1 - ALPHA / 2, df)
                            - float(stats.t.ppf(1 - ALPHA / 2, df)))
                        for df in range(1, 60))
            check("t quantile agrees with scipy", worst < 1e-10,
                  f"largest deviation {worst:.2e}")
        except ImportError:
            note("scipy not installed, the t quantile was not compared")

    cache = Path(args.cache)
    if not cache.exists():
        raise SystemExit(f"ABORT: {cache} missing. Run `karten` first.")
    d = pd.read_csv(cache)
    missing = [(a, f) for a in HEAD_ARMS for f in args.folds
               if not ((d["arm"] == a) & (d["fold"] == f)).any()]
    if missing:
        raise SystemExit(f"ABORT: the cache lacks {missing}. Run `karten` for "
                         f"those folds.")
    # The cache is a copy. A fold whose head field has since been archived or
    # rerun would still sit in it, and the report would then describe a run
    # that is no longer on disk. That is the same class of error as a file name
    # that no longer matches its contents.
    stale = [(a, f) for a in HEAD_ARMS for f in args.folds
             if not (Path(ARMS[a]) / f"head_f{f}_s{args.seed}.npz").exists()]
    if stale:
        raise SystemExit(
            "ABORT: the cache holds " + ", ".join(f"{a} fold {f}" for a, f in stale)
            + ", but the head field of that run is no longer on disk.\nRecompute "
              "it with `karten --folds ... --force` after the rerun, so the "
              "report describes the files that are actually there.")

    print("=" * 74)
    print("PHASE 5, THE SECOND HEAD")
    print("=" * 74)
    _, suspect = step0_provenance(args)
    winner, loser = step1_b(d, args)
    a = step2_a(args, winner, loser)
    step3_karten(d, args)
    step4_kreuzprobe(d, a, args)
    step5_sensitivitaet(d, a, suspect, winner, loser)

    print("\n" + "=" * 74)
    if FINDINGS:
        print(f"{len(FINDINGS)} FINDING(S):")
        for f in FINDINGS:
            print(f"  - {f}")
    else:
        print("No finding.")
    return 1 if FINDINGS else 0


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--cache", type=Path,
                       default=Path("predictions_p5_auswertung/karten_per_image.csv"))
        q.add_argument("--out-dir", type=Path,
                       default=Path("predictions_p5_auswertung"))

    k = sub.add_parser("karten", help="the expensive half, cached")
    common(k)
    k.add_argument("--arms", nargs="+", default=list(HEAD_ARMS))
    k.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    k.add_argument("--csv", type=Path, default=Path("data/rsna"))
    k.add_argument("--masks", type=Path, default=Path("data/rsna/masks224_dev"))
    k.add_argument("--baselines", type=Path,
                   default=Path("predictions_lokalisation"))
    k.add_argument("--force", action="store_true",
                   help="recompute folds that are already in the cache")
    k.set_defaults(func=run_karten)

    b = sub.add_parser("bericht", help="the statistics and the report")
    common(b)
    b.add_argument("--results", type=Path, default=Path("results_rsna.csv"))
    b.add_argument("--phase2", type=Path,
                   default=Path("predictions_cam_full/phase2_summary.csv"))
    b.add_argument("--offiziell", type=Path,
                   default=Path("predictions_p5_auswertung/offiziell_per_image.csv"),
                   help="per image output of rsna_kopf_auswertung.py, for the "
                        "second implementation check")
    b.add_argument("--selbsttest", action="store_true", default=True)
    b.add_argument("--batch", type=int, default=16,
                   help="batch size of the runs, for the lambda risk arithmetic")
    b.add_argument("--pos-share", type=float, default=0.225,
                   help="share of fitting images that carry a box")
    b.set_defaults(func=run_bericht)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
