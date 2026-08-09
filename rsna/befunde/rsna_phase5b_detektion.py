"""Phase 5b, Teil 2: Kaesten aus dem Kopffeld, IoU und mAP.

WHAT THIS PRODUCES
------------------
The RSNA competition metric for the two headed model, computed the way the
challenge computed it, with the location prior run through the same pipeline as
the null line.

  tune      searches the free parameters on the SELECTION split, per fold.
            Writes phase5b2_schwellen.csv, skips folds already in it.
  bericht   applies those parameters to the VALIDATION split and reports.

Nothing trains. Input is `head_sel_f*.npz` (tuning) and `head_f*.npz`
(reporting), both already on disk.

THE PIPELINE, PRESCRIBED BY THE ROADMAP
---------------------------------------
"aus einem Wahrscheinlichkeitsfeld werden Kaesten, indem man eine Schwelle
setzt, zusammenhaengende Kacheln verbindet und jedem Bereich sein Maximum als
Konfidenz gibt." Four steps, no second model:

  1. threshold the 14 by 14 field
  2. connected tiles, four neighbourhood
  3. bounding box of each region, in the 1024 grid the annotations use
  4. the region's maximum becomes its confidence

Regions below `min_tiles` are dropped. The boxes are coarse because the grid is
coarse: one tile is 1024/14 = 73 pixels. That is a property of the method and
belongs in every sentence about the result.

THE METRIC, TAKEN FROM THE COMPETITION AND NOT INVENTED HERE
------------------------------------------------------------
Eight IoU thresholds, 0.40 to 0.75 in steps of 0.05. At each one the
predictions are matched greedily to the annotations in order of descending
confidence, and the image scores

    TP / (TP + FN + FP)

The image's value is the mean over the eight thresholds, the reported number is
the mean over images. Two rules of the competition matter more than they look:

  * an image WITHOUT annotations that receives any prediction scores 0 and is
    counted. Crying wolf is punished directly.
  * an image without annotations and without predictions is EXCLUDED from the
    mean, not counted as a success.

Together these make the threshold a real trade off, and phase 5b part 1 says
why that matters here: at a cut of 0.5 the head fires on 62 percent of entirely
normal chests.

PRE-REGISTERED, WRITTEN BEFORE THE VALIDATION SPLIT WAS TOUCHED
---------------------------------------------------------------
Free parameters, all searched on the SELECTION split and per fold: field
threshold, `min_tiles`, and for E2 the classifier gate.

  E1, PRIMARY. The head alone. Boxes come from the field, nothing else is
      consulted. This is what phase 5 built, measured without help.

  E2, secondary. The chain. The classifier decides whether an image gets any
      boxes at all, the head decides where. This is what a deployed system
      does, and it is the number for the portfolio, labelled as a pipeline
      number and not as a property of the head.

  GATE, the only one. E1 has to beat the LOCATION PRIOR run through the same
      pipeline, paired per fold, 90 percent interval excluding zero. The prior
      is the map that knows nothing except where opacities usually sit; a
      detector that cannot beat it has measured the average chest and not the
      patient. If E1 fails, the competition numbers say nothing about the model
      and must not be quoted.

  The gate can discriminate, checked on the selection split of fold 0 before
  this was written: the metric runs from 0.0141 at a cut of 0.30 to 0.1302 at
  0.90 and back down to 0.0467 at 0.98. An interior maximum, no saturation at
  either end, far from zero. A 14 by 14 grid is therefore not too coarse for
  IoU 0.40, which was the real risk.

  Reported beside it, without a verdict: the same numbers per projection, the
  share of images that receive a box, the chosen parameters per fold, and what
  an untuned cut of 0.5 would have scored.

WHAT THIS NUMBER IS NOT
-----------------------
A leaderboard position. The competition scored a private test set with its own
case mix; this runs on the validation folds of this project, and the tuning
split is internal. The two are not comparable, and any sentence that puts them
side by side without saying so is wrong.

CLI, from the repository root:
  python rsna\\befunde\\rsna_phase5b_detektion.py tune --folds 0 1
  python rsna\\befunde\\rsna_phase5b_detektion.py bericht
"""

from __future__ import annotations

import _repo_path  # noqa: F401

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_lokalisation import load_boxes
from rsna_phase5_auswertung import ARMS, paired_t

GRID = 14
BOX_SPACE = 1024               # the grid the annotations live in
IOU_THR = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
ALPHA = 0.10
# The probe showed a smooth curve with an interior maximum near 0.90 for the
# head, so a step of 0.03 sits well inside the flat region around the optimum.
#
# The grid starts at 0.20 and not at 0.50 because of the NULL LINE, not because
# of the head. A first sweep from 0.50 put the location prior at the very edge
# of the grid, which means the search wanted to go lower and was not allowed
# to. A comparator that is held back by the grid is a comparator that is easy
# to beat, and the gate would then measure the grid rather than the model. The
# head never picks anything below 0.80, so the extra range costs time and
# nothing else.
FELD_GRID = np.round(np.arange(0.20, 0.981, 0.03), 3)
MIN_TILES = (1, 2, 3)
# Share of selection images the classifier gate may suppress. 0.0 means no
# gate, which keeps E1 inside the E2 sweep as a special case.
GATE_Q = np.round(np.arange(0.0, 0.91, 0.05), 3)
NAIV_THR = 0.5                 # the untuned cut, for the descriptive line

FINDINGS: list[str] = []


def check(name: str, cond: bool, info: str = "") -> bool:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FINDINGS.append(name)
    return cond


def note(text: str) -> None:
    print(f"        {text}")


# --------------------------------------------------------------------------
# Field to boxes
# --------------------------------------------------------------------------

def komponenten(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Connected tiles, four neighbourhood, written out rather than imported.

    scipy would do this in one call, and scipy is what this project does not
    want to depend on for a number that has to be recomputable on a plain
    interpreter. On 196 tiles a flood fill costs nothing.
    """
    seen = np.zeros_like(mask, bool)
    out = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                comp = []
                while q:
                    y, x = q.popleft()
                    comp.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        b, a = y + dy, x + dx
                        if 0 <= b < h and 0 <= a < w and mask[b, a] and not seen[b, a]:
                            seen[b, a] = True
                            q.append((b, a))
                out.append(comp)
    return out


def regionen(field: np.ndarray, thr: float) -> list[tuple[tuple, float, int]]:
    """(box, score, tile count) per region. Extracted ONCE per threshold.

    `min_tiles` filters this list afterwards instead of forcing a second flood
    fill per value, which is the difference between a sweep of minutes and one
    of half an hour.
    """
    if float(field.max()) < thr:
        return []
    s = BOX_SPACE / GRID
    out = []
    for comp in komponenten(field >= thr):
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]
        box = (min(xs) * s, min(ys) * s,
               (max(xs) - min(xs) + 1) * s, (max(ys) - min(ys) + 1) * s)
        out.append((box, max(float(field[y, x]) for y, x in comp), len(comp)))
    return out


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def map_iou(bt: np.ndarray, bp: np.ndarray, sc: np.ndarray):
    """The competition metric for one image. None means "not counted at all"."""
    if len(bt) == 0 and len(bp) == 0:
        return None
    if len(bp):
        bp = bp[np.argsort(sc)[::-1], :]
    total = 0.0
    for t in IOU_THR:
        matched = set()
        tp = fn = 0
        for b in bt:
            hit = False
            for j, p in enumerate(bp):
                if not hit and j not in matched and iou(b, p) >= t:
                    hit = True
                    tp += 1
                    matched.add(j)
            if not hit:
                fn += 1
        fp = len(bp) - len(matched)
        total += tp / (tp + fn + fp) if (tp + fn + fp) else 0.0
    return total / len(IOU_THR)


def je_bild(fields, ids, boxes, thr, konstant=False):
    """Per image and per `min_tiles`: metric, annotations present, boxes given.

    `konstant=True` says every image sees the SAME field, which is true for the
    location prior. The regions are then extracted once instead of n times.
    """
    res = {mt: ([], [], []) for mt in MIN_TILES}
    festen = regionen(fields[0], thr) if konstant else None
    for k, pid in enumerate(ids):
        regs = festen if konstant else regionen(fields[k], thr)
        bt = np.asarray(boxes.get(pid, []), float).reshape(-1, 4)
        for mt in MIN_TILES:
            keep = [r for r in regs if r[2] >= mt]
            bp = np.asarray([r[0] for r in keep], float).reshape(-1, 4)
            sc = np.asarray([r[1] for r in keep], float)
            m, g, n = res[mt]
            v = map_iou(bt, bp, sc)
            m.append(np.nan if v is None else v)
            g.append(len(bt) > 0)
            n.append(len(bp))
    return {mt: (np.array(a), np.array(b), np.array(c))
            for mt, (a, b, c) in res.items()}


def score_mit_tor(m, has_gt, n_pred, p, gate):
    """The metric once a classifier gate suppresses images below `gate`.

    A suppressed image predicts nothing. It scores 0 if it carries annotations
    and drops out of the mean if it does not, exactly as the competition rule
    says. Nothing is recomputed, the gate only reselects, which is why the
    two dimensional search costs the same as the one dimensional one.
    """
    an = p >= gate
    werte = np.where(an, m, np.where(has_gt, 0.0, np.nan))
    werte = werte[np.isfinite(werte)]
    return (float(werte.mean()) if len(werte) else float("nan"),
            int(len(werte)), float((an & (n_pred > 0)).mean()))


# --------------------------------------------------------------------------

def lade_feld(pfad: Path):
    z = np.load(pfad, allow_pickle=False)
    return [str(x) for x in z["patientId"]], z["field"]


def prior_feld(fold: int, n: int, args) -> np.ndarray:
    """The location prior on the head grid, the same map for every image.

    Scaled to a maximum of 1 so the SAME threshold grid means something for
    both sources. Without that the comparison would be between a map and a
    scale, not between two maps.
    """
    p = np.load(Path(args.baselines) / f"prior_f{fold}.npy")
    k = p.shape[0] // GRID
    grob = p.reshape(GRID, k, GRID, k).mean(axis=(1, 3))
    grob = grob / max(float(grob.max()), 1e-12)
    return np.repeat(grob[None, :, :], max(n, 1), axis=0)


def run_tune(args) -> int:
    boxes = load_boxes(args.csv)
    out = Path(args.out_dir) / "phase5b2_schwellen.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    alt = pd.read_csv(out) if out.exists() else pd.DataFrame()

    rows = []
    for fold in args.folds:
        f = Path(ARMS[args.arm]) / f"head_sel_f{fold}_s{args.seed}.npz"
        if not f.exists():
            raise SystemExit(f"ABORT: {f} missing. Run rsna_kopf_sel.py first, "
                             f"the tuning data is not on disk.")
        ids, kopf = lade_feld(f)
        sel = pd.read_csv(Path(ARMS[args.arm])
                          / f"sel_f{fold}_s{args.seed}.csv").set_index("patientId")
        p = sel.loc[ids, "p_sel"].to_numpy()

        for quelle in ("kopf", "priore"):
            fertig = (not alt.empty
                      and ((alt["fold"] == fold) & (alt["quelle"] == quelle)).any())
            if fertig and not args.force:
                print(f"  fold {fold} {quelle}: already tuned, skipped")
                continue
            fields = kopf if quelle == "kopf" else prior_feld(fold, len(ids), args)
            konstant = quelle == "priore"
            best = {"E1": None, "E2": None}
            for thr in FELD_GRID:
                per_mt = je_bild(fields, ids, boxes, float(thr), konstant)
                for mt in MIN_TILES:
                    m, g, n = per_mt[mt]
                    for q in GATE_Q:
                        gate = float(np.quantile(p, q)) if q > 0 else -1.0
                        s, nn, anteil = score_mit_tor(m, g, n, p, gate)
                        cand = {"fold": fold, "quelle": quelle,
                                "thr": float(thr), "min_tiles": int(mt),
                                "gate_q": float(q), "gate": gate,
                                "score_sel": s, "n_sel": nn,
                                "anteil_sel": anteil}
                        for arm in (("E1", "E2") if q == 0.0 else ("E2",)):
                            if best[arm] is None or s > best[arm]["score_sel"]:
                                best[arm] = dict(cand, endpunkt=arm)
                print(f"    fold {fold} {quelle}: thr {thr:.2f} ", end="\r")
            for arm in ("E1", "E2"):
                rows.append(best[arm])
                b = best[arm]
                print(f"\n  fold {fold} {quelle} {arm}: thr {b['thr']:.2f}, "
                      f"min_tiles {b['min_tiles']}, gate_q {b['gate_q']:.2f}, "
                      f"sel {b['score_sel']:.4f}")
            # Nach JEDER Quelle schreiben, nicht erst am Ende. Ein Abbruch
            # mitten im Durchlauf soll die schon gesuchten Folds nicht kosten,
            # und das Ueberspringen oben funktioniert nur, wenn sie auf der
            # Platte stehen.
            alt = _schreiben(out, alt, rows)
            rows = []

    if rows:
        _schreiben(out, alt, rows)
    return 0


def _schreiben(out: Path, alt: pd.DataFrame, rows: list) -> pd.DataFrame:
    if not rows:
        return alt
    neu = pd.DataFrame(rows)
    d = pd.concat([alt, neu], ignore_index=True) if not alt.empty else neu
    d = d.drop_duplicates(subset=["fold", "quelle", "endpunkt"], keep="last")
    d.to_csv(out, index=False)
    print(f"    {len(d)} Zeilen in {out}")
    return d


# --------------------------------------------------------------------------

def val_werte(fold, quelle, endpunkt, par, boxes, args):
    """Apply one tuned parameter set to the validation split of one fold."""
    f = Path(ARMS[args.arm]) / f"head_f{fold}_s{args.seed}.npz"
    ids, kopf = lade_feld(f)
    val = pd.read_csv(Path(ARMS[args.arm])
                      / f"rsna_f{fold}_s{args.seed}.csv").set_index("patientId")
    p = val.loc[ids, "p_clean"].to_numpy()
    vp = val.loc[ids, "viewpos"].to_numpy().astype(str)
    fields = kopf if quelle == "kopf" else prior_feld(fold, len(ids), args)
    per_mt = je_bild(fields, ids, boxes, float(par["thr"]), quelle == "priore")
    m, g, n = per_mt[int(par["min_tiles"])]
    # The gate is a QUANTILE of the selection scores, so it is recomputed on
    # the selection distribution and applied here. Taking the quantile of the
    # validation scores instead would let the reporting set set its own cut.
    gate = float(par["gate"])
    return m, g, n, p, vp, gate


def run_bericht(args) -> int:
    boxes = load_boxes(args.csv)
    pfad = Path(args.out_dir) / "phase5b2_schwellen.csv"
    if not pfad.exists():
        raise SystemExit(f"ABORT: {pfad} missing. Run `tune` first.")
    par = pd.read_csv(pfad)

    print("=" * 74)
    print("PHASE 5b, TEIL 2: DETEKTION, IoU UND mAP")
    print("=" * 74)

    print("\n0  DIE PARAMETER, ALLE AUF DEM SELEKTIONS-SPLIT BESTIMMT")
    fehlt = [(f, q, e) for f in args.folds for q in ("kopf", "priore")
             for e in ("E1", "E2")
             if not ((par.fold == f) & (par.quelle == q) & (par.endpunkt == e)).any()]
    if fehlt:
        raise SystemExit(f"ABORT: no tuned parameters for {fehlt}. Run `tune`.")
    print(par.sort_values(["quelle", "endpunkt", "fold"])[
        ["quelle", "endpunkt", "fold", "thr", "min_tiles", "gate_q",
         "score_sel"]].to_string(index=False))
    check("kein Parametersatz sitzt am Rand des Suchgitters",
          bool((par.thr > FELD_GRID.min()).all() and (par.thr < FELD_GRID.max()).all()),
          f"thr von {par.thr.min():.2f} bis {par.thr.max():.2f}, Gitter "
          f"{FELD_GRID.min():.2f} bis {FELD_GRID.max():.2f}")

    zeilen = []
    for quelle in ("kopf", "priore"):
        for endpunkt in ("E1", "E2"):
            for fold in args.folds:
                p_row = par[(par.fold == fold) & (par.quelle == quelle)
                            & (par.endpunkt == endpunkt)].iloc[-1]
                m, g, n, p, vp, gate = val_werte(fold, quelle, endpunkt,
                                                 p_row, boxes, args)
                s, nn, anteil = score_mit_tor(m, g, n, p, gate)
                r = {"quelle": quelle, "endpunkt": endpunkt, "fold": fold,
                     "score": s, "n": nn, "anteil_mit_kasten": anteil}
                for v in ("AP", "PA"):
                    msk = vp == v
                    sv, _, _ = score_mit_tor(m[msk], g[msk], n[msk], p[msk], gate)
                    r[f"score_{v}"] = sv
                zeilen.append(r)
    d = pd.DataFrame(zeilen)
    d.to_csv(Path(args.out_dir) / "phase5b2_je_fold.csv", index=False)

    print("\n" + "=" * 74)
    print("1  DAS WETTBEWERBSMASS AUF DER VALIDIERUNG")
    print("=" * 74)
    print(f"\n  {'Quelle':>7} {'Endp':>5} " + " ".join(f"{f'f{f}':>7}" for f in args.folds)
          + f" {'Mittel':>8} {'nur AP':>8} {'nur PA':>8} {'mit Kasten':>11}")
    for quelle in ("kopf", "priore"):
        for endpunkt in ("E1", "E2"):
            s = d[(d.quelle == quelle) & (d.endpunkt == endpunkt)].set_index("fold")
            print(f"  {quelle:>7} {endpunkt:>5} "
                  + " ".join(f"{s.loc[f, 'score']:>7.4f}" for f in args.folds)
                  + f" {s['score'].mean():>8.4f} {s['score_AP'].mean():>8.4f} "
                    f"{s['score_PA'].mean():>8.4f} "
                    f"{s['anteil_mit_kasten'].mean():>11.3f}")

    print("\n" + "=" * 74)
    print("2  DAS TOR: DER KOPF GEGEN DEN LAGEPRIORE, GEPAART JE FOLD")
    print("=" * 74)
    for endpunkt in ("E1", "E2"):
        k = d[(d.quelle == "kopf") & (d.endpunkt == endpunkt)].set_index("fold")["score"]
        l = d[(d.quelle == "priore") & (d.endpunkt == endpunkt)].set_index("fold")["score"]
        r = paired_t((k - l).to_numpy(), ALPHA)
        marke = "BESTAETIGEND" if endpunkt == "E1" else "erkundend"
        print(f"\n  {endpunkt} ({marke}): Kopf {k.mean():.4f}, Lagepriore "
              f"{l.mean():.4f}, Differenz {r['mean']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}]")
        if endpunkt == "E1":
            if r["lo"] > 0:
                print("  TOR BESTANDEN. Der Kopf schlaegt die Karte, die nur "
                      "die uebliche Lage kennt.")
            else:
                print("  TOR DURCHGEFALLEN. Die Wettbewerbszahlen sagen dann "
                      "nichts ueber das Modell")
                print("  aus und duerfen nicht zitiert werden.")
                FINDINGS.append("E1 schlaegt den Lagepriore nicht gesichert")

    print("\n" + "=" * 74)
    print("3  WAS EINE UNGETUNTE SCHWELLE GEKOSTET HAETTE")
    print("=" * 74)
    print("  Beschreibend. Schwelle 0,5, kein Tor, min_tiles 1, also der Kopf")
    print("  so wie er aus dem Training kommt.\n")
    naiv = []
    for fold in args.folds:
        ids, kopf = lade_feld(Path(ARMS[args.arm])
                              / f"head_f{fold}_s{args.seed}.npz")
        per_mt = je_bild(kopf, ids, boxes, NAIV_THR, False)
        m, g, n = per_mt[1]
        s, _, anteil = score_mit_tor(m, g, n, np.ones(len(ids)), -1.0)
        naiv.append({"fold": fold, "score": s, "anteil": anteil})
    nd = pd.DataFrame(naiv)
    getunt = d[(d.quelle == "kopf") & (d.endpunkt == "E1")]["score"].mean()
    print(f"  ungetunt: {nd['score'].mean():.4f}   "
          f"(Kasten auf {nd['anteil'].mean():.1%} der Bilder)")
    print(f"  getunt:   {getunt:.4f}")
    note("Der Unterschied ist die Kalibrierung, die die Schwellenwahl auf dem")
    note("Selektions-Split nebenbei miterledigt, siehe Phase 5b Teil 1.")

    print("\n" + "=" * 74)
    if FINDINGS:
        print(f"{len(FINDINGS)} BEFUND(E):")
        for f in FINDINGS:
            print(f"  - {f}")
    else:
        print("Kein Befund.")
    return 1 if FINDINGS else 0


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--arm", default="ex", choices=sorted(ARMS))
        q.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--csv", type=Path, default=Path("data/rsna"))
        q.add_argument("--baselines", type=Path,
                       default=Path("predictions_lokalisation"))
        q.add_argument("--out-dir", type=Path,
                       default=Path("predictions_p5_auswertung"))

    t = sub.add_parser("tune", help="die freien Parameter, auf dem Selektions-Split")
    common(t)
    t.add_argument("--force", action="store_true")
    t.set_defaults(func=run_tune)

    b = sub.add_parser("bericht", help="die Zahlen auf der Validierung")
    common(b)
    b.set_defaults(func=run_bericht)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
