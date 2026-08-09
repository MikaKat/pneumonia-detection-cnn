"""Phase 5b, Teil 1: schlaegt der Kopf auf Bildern ohne Pneumonie Alarm.

WHAT THIS PRODUCES
------------------
The false alarm behaviour of the localisation head, broken down by the three
RSNA classes. Nothing here trains and nothing needs a GPU: the head fields of
every validation image are already on disk, negatives included.

  raster    the only slow part, one pass over the lung masks to record which
            of the 14 by 14 tiles are lung. Cached per fold.
  bericht   the endpoints, the gate and the tables. Seconds.

WHY IT EXISTS
-------------
The head of the winning arm was trained with `--head-negatives exclude`. Under
that recipe an image without a box contributes nothing to the head loss, so the
head has never seen a film on which it was supposed to stay quiet. Everything
phase 5 measured was measured on annotated images, which is the half of the
question where that cannot show.

The other half matters more than it first looks, because of how RSNA is built:

    Lung Opacity (box, label 1)          22.5 %
    No Lung Opacity / Not Normal         44.3 %      <- the largest class
    Normal                               33.2 %

The largest class is the middle one: films that show something which is not
pneumonia. Effusion, congestion, scarring, cardiomegaly, a device. The head has
never seen one. That is exactly where a pneumonia overlay would be tempted to
light up, and phase 3 already found the classifier's false positives sitting
there (alarm rate 0.373 in the middle class against 0.019 in Normal).

Phase 3 left this open in writing: "whether the heat map points at the other
pathology that is actually present" could not be answered, because it would
have needed Grad-CAM on negative images. The head field costs nothing here, so
the question can be closed now.

PRE-REGISTERED, WRITTEN BEFORE ANY CLASS WAS LOOKED AT
------------------------------------------------------
The image level score is the ALARM: the maximum of the head field over the
tiles that are lung. A tile counts as lung when more than half of its area is
inside the lung mask. Chosen before the breakdown, on pooled data alone, for
one reason: it has to have room to move. Measured on fold 0, 600 images, no
class labels involved: median 0.84, quartiles 0.55 and 0.96, 9 percent above
0.99. Compressed at the top but far from saturated, and AUC is a rank measure
so compression costs nothing. The mean over the same tiles is carried along as
the EXTENT of the alarm; it separates "fires hard in one spot" from "fires
weakly everywhere".

  D1, PRIMARY and the only gate. AUC(alarm) separating Lung Opacity from the
      MIDDLE class, per fold, 90 percent interval over the five folds.

      GATE: the interval has to lie entirely above 0.5. Above 0.5 means the
      head distinguishes pneumonia from other abnormality. At 0.5 it is a
      generic opacity detector, and then the negatives have to enter the head
      loss, which is the alpha dial between `exclude` and `empty`.

      The gate can discriminate, which is checked and not assumed: the score
      has the spread quoted above, and every fold holds roughly 1030 opacity
      and 2040 middle class images, so neither end is starved.

  D2  the same AUC against NORMAL, beside it. Expectation, written down so it
      can be wrong: higher than against the middle class. If it is not, the
      reading of D1 is broken and nothing else in this file should be believed.

  D3  the three class ordering of the mean alarm. Expected Normal below Middle
      below Opacity.

  D4  the same two AUCs for the CLASSIFIER score on the same images, as the
      reference line. Descriptive, no verdict: no margin was ever pre-registered
      for this comparison and inventing one now would be a gate added after the
      fact.

  D5  `exclude` against `empty` on D1, paired per fold. This is the number that
      says whether the alpha dial is worth building: `empty` saw the negatives,
      `exclude` did not.

  D6  descriptive, closing the phase 3 loose end: does the peak of the head sit
      inside the lung field at all, on the images where it fires wrongly.

Everything is broken down by projection as well. That rule was paid for in
phase 3, where every pooled headline fell apart per stratum.

WHAT IS NOT IN HERE
-------------------
The detection numbers of the roadmap, IoU and mAP. They need a threshold, and
the roadmap says the threshold comes from the SELECTION split. The head fields
of the selection split are not on disk, only those of the validation split.
Setting the threshold on the validation split instead would break a rule that
was written down before, so that part waits until a forward pass over the
selection split exists. It is cheap, but it is compute and it is not this file.

CLI, from the repository root:
  python rsna\\befunde\\rsna_phase5b_falschalarm.py raster --folds 0 1
  python rsna\\befunde\\rsna_phase5b_falschalarm.py bericht
"""

from __future__ import annotations

import _repo_path  # noqa: F401

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_lokalisation import REF_SIZE, load_lung
from rsna_phase5_auswertung import ARMS, HEAD_ARMS, paired_t, rank_auc

GRID = 14
TILE = REF_SIZE // GRID           # 16 pixels per tile at 224
LUNG_TILE = 0.5                   # a tile is lung when more than half of it is
ALPHA = 0.10

KLASSEN = {"Lung Opacity": "Infiltrat",
           "No Lung Opacity / Not Normal": "Mittelklasse",
           "Normal": "Normal"}
ORDER = ["Normal", "Mittelklasse", "Infiltrat"]

FINDINGS: list[str] = []


def check(name: str, cond: bool, info: str = "") -> bool:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FINDINGS.append(name)
    return cond


def note(text: str) -> None:
    print(f"        {text}")


# --------------------------------------------------------------------------
# The slow half: which tiles are lung
# --------------------------------------------------------------------------

def run_raster(args) -> int:
    sp = json.loads(Path(args.splits).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fold in args.folds:
        f = out_dir / f"lungenraster_f{fold}.npz"
        if f.exists() and not args.force:
            print(f"  fold {fold}: {f} exists, skipped")
            continue
        val_ids = sp["folds"][fold]["val"]
        cov = np.zeros((len(val_ids), GRID, GRID), np.float32)
        fehlt = []

        def eine(k_pid):
            k, pid = k_pid
            lung = load_lung(args.masks, pid)
            if lung is None:
                return k, None
            # 224 = 14 x 16, so the tile grid divides the image exactly and the
            # coverage of a tile is the mean of its 16 by 16 block. No
            # resampling, therefore no interpolation artefact at the lung edge.
            return k, np.asarray(lung, np.float32).reshape(
                GRID, TILE, GRID, TILE).mean(axis=(1, 3))

        # Threads, not processes: this loop waits on the disk, it does not
        # compute. The arithmetic per image is a mean over 196 blocks and takes
        # microseconds, opening the PNG takes milliseconds. Order is restored
        # through the index, so the result does not depend on the scheduling.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for done, (k, c) in enumerate(pool.map(eine, enumerate(val_ids)), 1):
                if c is None:
                    fehlt.append(val_ids[k])
                else:
                    cov[k] = c
                if done % 2000 == 0:
                    print(f"    {done}/{len(val_ids)}")
        np.savez_compressed(f, patientId=np.array(val_ids), coverage=cov)
        print(f"  fold {fold}: {len(val_ids)} images, {len(fehlt)} without a "
              f"lung mask -> {f}")
    return 0


# --------------------------------------------------------------------------

def load_klassen(csv_dir: Path) -> pd.Series:
    d = pd.read_csv(Path(csv_dir) / "stage_2_detailed_class_info.csv")
    d = d.drop_duplicates("patientId")
    return d.set_index("patientId")["class"].map(KLASSEN)


def alarm_tabelle(args) -> pd.DataFrame:
    """One row per image and arm: alarm, extent, class, projection."""
    sp = json.loads(Path(args.splits).read_text())
    klasse = load_klassen(args.csv)
    vpmap = sp["viewpos"]
    rows = []
    for arm in HEAD_ARMS:
        for fold in args.folds:
            zf = Path(ARMS[arm]) / f"head_f{fold}_s{args.seed}.npz"
            rf = Path(args.out_dir) / f"lungenraster_f{fold}.npz"
            if not zf.exists():
                raise SystemExit(f"ABORT: {zf} missing.")
            if not rf.exists():
                raise SystemExit(f"ABORT: {rf} missing. Run `raster` first.")
            z = np.load(zf, allow_pickle=False)
            r = np.load(rf, allow_pickle=False)
            ids = [str(x) for x in z["patientId"]]
            if ids != [str(x) for x in r["patientId"]]:
                raise SystemExit(
                    f"ABORT: {zf} and {rf} hold different images or a different "
                    f"order. One of them belongs to another split.")
            field, cov = z["field"], r["coverage"]
            tiles = cov > LUNG_TILE
            for k, pid in enumerate(ids):
                t = tiles[k]
                if not t.any():
                    continue
                f = field[k]
                rows.append({
                    "arm": arm, "fold": fold, "patientId": pid,
                    "klasse": klasse.get(pid, "?"),
                    "viewpos": vpmap.get(pid, "?"),
                    "alarm": float(f[t].max()),
                    "ausdehnung": float(f[t].mean()),
                    "alarm_bild": float(f.max()),
                    "lungenkacheln": int(t.sum()),
                })
    return pd.DataFrame(rows)


def auc_je_fold(d: pd.DataFrame, spalte: str, pos: str, neg: str) -> pd.Series:
    """AUC separating class `pos` from class `neg`, one value per fold."""
    out = {}
    for fold, g in d.groupby("fold"):
        s = g[g["klasse"].isin((pos, neg))]
        if s["klasse"].nunique() < 2:
            continue
        out[fold] = rank_auc((s["klasse"] == pos).to_numpy().astype(int),
                             s[spalte].to_numpy())
    return pd.Series(out)


def zeile(label: str, a: pd.Series) -> dict:
    r = paired_t((a - 0.5).to_numpy(), ALPHA)
    print(f"  {label:<34} AUC {a.mean():.4f}   90 % [{0.5 + r['lo']:.4f}, "
          f"{0.5 + r['hi']:.4f}]   je Fold "
          + " ".join(f"{v:.3f}" for v in a.sort_index()))
    return {"vergleich": label, "auc": float(a.mean()),
            "lo": 0.5 + r["lo"], "hi": 0.5 + r["hi"]}


# --------------------------------------------------------------------------

def run_bericht(args) -> int:
    d = alarm_tabelle(args)
    out_dir = Path(args.out_dir)
    d.to_csv(out_dir / "phase5b_alarm_per_image.csv", index=False)

    print("=" * 74)
    print("PHASE 5b, TEIL 1: FALSCHALARM DES KOPFES")
    print("=" * 74)

    print("\n0  WAS UEBERHAUPT GEMESSEN WIRD")
    n = d[d.arm == "ex"].groupby(["fold", "klasse"]).size().unstack(fill_value=0)
    print(n.to_string())
    check("jede Klasse ist in jedem Fold ausreichend besetzt",
          bool((n[ORDER] >= 100).all().all()),
          f"kleinste Gruppe {int(n[ORDER].min().min())}")
    check("keine unbekannte Klasse",
          not (d["klasse"] == "?").any(),
          f"{int((d['klasse'] == '?').sum())} Bilder ohne Klasse")
    ohne = d[d.arm == "ex"]["lungenkacheln"]
    print(f"  Lungenkacheln je Bild: Median {ohne.median():.0f} von "
          f"{GRID * GRID}, Minimum {ohne.min()}")

    ergebnisse = []
    for arm in HEAD_ARMS:
        s = d[d.arm == arm]
        print("\n" + "=" * 74)
        print(f"ARM {arm}")
        print("=" * 74)

        print("\nD3  Alarm je Klasse (Mittel), erwartete Reihenfolge "
              "Normal < Mittelklasse < Infiltrat")
        t = s.groupby("klasse")[["alarm", "ausdehnung"]].mean().reindex(ORDER)
        t["n"] = s.groupby("klasse").size().reindex(ORDER)
        print(t.round(4).to_string())
        ok_order = bool(t["alarm"].is_monotonic_increasing)
        check(f"{arm}: die Reihenfolge stimmt", ok_order)

        print("\nD1 und D2  Trennung durch den Alarm, gepaart je Fold")
        r1 = zeile("D1 Infiltrat gegen Mittelklasse",
                   auc_je_fold(s, "alarm", "Infiltrat", "Mittelklasse"))
        r2 = zeile("D2 Infiltrat gegen Normal",
                   auc_je_fold(s, "alarm", "Infiltrat", "Normal"))
        r3 = zeile("   Mittelklasse gegen Normal",
                   auc_je_fold(s, "alarm", "Mittelklasse", "Normal"))
        for r in (r1, r2, r3):
            r["arm"] = arm
            ergebnisse.append(r)

        # D1 je Projektion und daraus die geschichtete Zahl, genau wie bei
        # Endpunkt A. Die gepoolte AUC traegt den Projektionseffekt mit, sie ist
        # deshalb systematisch die freundlichere. Welche der beiden man liest,
        # ist in diesem Projekt keine Geschmacksfrage mehr.
        print("\n  je Projektion, und die geschichtete Zahl daneben")
        teile, gew = [], []
        for v in ("AP", "PA"):
            sv = s[s["viewpos"] == v]
            a = auc_je_fold(sv, "alarm", "Infiltrat", "Mittelklasse")
            n = int(sv["klasse"].isin(("Infiltrat", "Mittelklasse")).sum())
            if len(a):
                teile.append(a)
                gew.append(n)
                print(f"    nur {v}: D1 = {a.mean():.4f}   n {n}   je Fold "
                      + " ".join(f"{x:.3f}" for x in a.sort_index()))
        if len(teile) == 2:
            w = np.asarray(gew, float)
            strat = (teile[0] * w[0] + teile[1] * w[1]) / w.sum()
            rs = paired_t((strat - 0.5).to_numpy(), ALPHA)
            print(f"    GESCHICHTET: {strat.mean():.4f}   90 % "
                  f"[{0.5 + rs['lo']:.4f}, {0.5 + rs['hi']:.4f}]   "
                  f"gepoolt {r1['auc']:.4f}")
            ergebnisse.append({"arm": arm, "vergleich": "D1 geschichtet",
                               "auc": float(strat.mean()),
                               "lo": 0.5 + rs["lo"], "hi": 0.5 + rs["hi"]})
            if strat.mean() < r1["auc"] - 0.01:
                note(f"Die gepoolte Zahl liegt {r1['auc'] - strat.mean():+.4f} "
                     f"ueber der geschichteten. Berichtet wird die")
                note("geschichtete, alles andere waere der Fehler aus Phase 3.")

        print("\n  DAS TOR")
        if r1["lo"] > 0.5:
            print(f"  BESTANDEN. Das Intervall liegt ganz ueber 0,5. Der Kopf "
                  f"unterscheidet")
            print("  Pneumonie von anderer Auffaelligkeit, er ist kein reiner "
                  "Verschattungsmelder.")
        else:
            print("  DURCHGEFALLEN. Das Intervall schliesst 0,5 ein. Auf der "
                  "Mittelklasse")
            print("  verhaelt sich der Kopf wie auf Infiltraten. Dann muessen "
                  "die Negativen in")
            print("  den Kopfverlust, und der naechste Versuch ist der "
                  "alpha-Regler.")
            FINDINGS.append(f"{arm}: D1 schliesst 0,5 ein")
        if not (r2["auc"] > r1["auc"]):
            note("D2 liegt NICHT ueber D1. Die Erwartung war umgekehrt, und")
            note("solange das nicht geklaert ist, ist die Lesart von D1 unklar.")
            FINDINGS.append(f"{arm}: D2 liegt nicht ueber D1")

    print("\n" + "=" * 74)
    print("D4  DIE BEZUGSLINIE: DER KLASSIFIKATOR AUF DENSELBEN BILDERN")
    print("=" * 74)
    print("  Beschreibend, ohne Urteil: fuer diesen Vergleich wurde nie eine")
    print("  Marge vorfestgelegt, und eine jetzt zu erfinden waere ein "
          "nachtraegliches Tor.\n")
    kl = load_klassen(args.csv)
    for arm in HEAD_ARMS:
        parts = []
        for fold in args.folds:
            p = Path(ARMS[arm]) / f"rsna_f{fold}_s{args.seed}.csv"
            t = pd.read_csv(p)
            t["fold"] = fold
            t["klasse"] = t["patientId"].map(kl)
            parts.append(t)
        c = pd.concat(parts, ignore_index=True)
        a1 = auc_je_fold(c, "p_clean", "Infiltrat", "Mittelklasse")
        a2 = auc_je_fold(c, "p_clean", "Infiltrat", "Normal")
        print(f"  {arm}, Klassifikatorwert:  Infiltrat gegen Mittelklasse "
              f"{a1.mean():.4f},  gegen Normal {a2.mean():.4f}")
        kopf1 = next(r["auc"] for r in ergebnisse
                     if r["arm"] == arm and r["vergleich"].startswith("D1"))
        diff = paired_t((auc_je_fold(d[d.arm == arm], "alarm", "Infiltrat",
                                     "Mittelklasse") - a1).to_numpy(), ALPHA)
        print(f"      Kopf minus Klassifikator auf D1: {diff['mean']:+.4f} "
              f"[{diff['lo']:+.4f}, {diff['hi']:+.4f}]   "
              f"(Kopf {kopf1:.4f})")

    print("\n" + "=" * 74)
    print("D5  EXCLUDE GEGEN EMPTY AUF D1, gepaart je Fold")
    print("=" * 74)
    print("  Die Zahl, die sagt, ob der alpha-Regler gebaut werden muss.")
    a_ex = auc_je_fold(d[d.arm == "ex"], "alarm", "Infiltrat", "Mittelklasse")
    a_em = auc_je_fold(d[d.arm == "em"], "alarm", "Infiltrat", "Mittelklasse")
    r = paired_t((a_ex - a_em).to_numpy(), ALPHA)
    print(f"  ex {a_ex.mean():.4f}   em {a_em.mean():.4f}   "
          f"Differenz {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
    if r["hi"] < 0:
        print("  `empty` ist auf der Mittelklasse gesichert spezifischer. Genau")
        print("  der Zielkonflikt, auf dem der alpha-Regler sitzen wuerde: "
              "`exclude`")
        print("  zeigt genauer, `empty` alarmiert seltener falsch.")
        FINDINGS.append("empty ist auf der Mittelklasse spezifischer als exclude")
    elif r["lo"] > 0:
        print("  `exclude` ist auch hier besser. Dann kostet das Weglassen der")
        print("  Negativen nichts und der Regler waere eine Neugier, kein "
              "Versuch.")
    else:
        print("  Nicht getrennt. Das heisst NICHT 'kein Unterschied', sondern")
        print(f"  'zu ungenau gemessen': Halbbreite {r['half']:.4f} bei fuenf "
              f"Folds.")

    print("\n" + "=" * 74)
    print("D6  LIEGT DER AUSSCHLAG UEBERHAUPT IM LUNGENFELD")
    print("=" * 74)
    print("  Die Frage, die Phase 3 offen lassen musste, weil sie Grad-CAM auf")
    print("  negativen Bildern gebraucht haette.\n")
    for arm in HEAD_ARMS:
        s = d[d.arm == arm]
        for k in ORDER:
            sub = s[s["klasse"] == k]
            drin = float(np.isclose(sub["alarm"], sub["alarm_bild"]).mean())
            print(f"  {arm} {k:<13} Gipfel in der Lunge bei {drin:.3f} der "
                  f"Bilder   (Alarm Lunge {sub['alarm'].mean():.4f}, "
                  f"ganzes Bild {sub['alarm_bild'].mean():.4f})")

    print("\n" + "=" * 74)
    print("D7  WIE LAUT IST DER KOPF ABSOLUT")
    print("=" * 74)
    print("  NICHT vorfestgelegt, beschreibend. AUC ist eine Rangzahl und sagt")
    print("  ueber die Hoehe nichts. Sobald irgendwo eine feste Schwelle auf das")
    print("  Kopffeld gelegt wird, und Phase 5b Teil 2 braucht genau das, zaehlt")
    print("  aber die Hoehe. Anteil der Bilder mit einem Alarm ueber der "
          "Schwelle:\n")
    print(f"  {'Arm':>3} {'Schwelle':>9} {'Normal':>9} {'Mittelklasse':>13} "
          f"{'Infiltrat':>10}")
    laut = {}
    for arm in HEAD_ARMS:
        s = d[d.arm == arm]
        for thr in (0.5, 0.8, 0.9):
            q = [float((s[s["klasse"] == k]["alarm"] > thr).mean())
                 for k in ORDER]
            if thr == 0.5:
                laut[arm] = q[0]
            print(f"  {arm:>3} {thr:>9.2f} {q[0]:>9.3f} {q[1]:>13.3f} "
                  f"{q[2]:>10.3f}")
    print()
    note("Dieselbe Form wie in Phase 3: die Rangfolge stimmt, die Hoehe nicht.")
    note("Dort war die Antwort Platt auf dem Selektions-Split und nicht ein")
    note("neues Training, und dieselbe Antwort liegt hier naeher als der")
    note("alpha-Regler, solange D5 die beiden Arme nicht trennt.")
    FINDINGS.append(
        f"nicht vorfestgelegt, aber unuebersehbar: bei Schwelle 0,5 alarmiert "
        f"der Kopf auf {laut.get('ex', float('nan')):.1%} der NORMALEN Bilder "
        f"in `exclude` und auf {laut.get('em', float('nan')):.1%} in `empty`")

    pd.DataFrame(ergebnisse).to_csv(out_dir / "phase5b_urteil.csv", index=False)

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
        q.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
        q.add_argument("--csv", type=Path, default=Path("data/rsna"))
        q.add_argument("--out-dir", type=Path,
                       default=Path("predictions_p5_auswertung"))

    r = sub.add_parser("raster", help="Lungenkacheln je Bild, einmalig")
    common(r)
    r.add_argument("--masks", type=Path, default=Path("data/rsna/masks224_dev"))
    r.add_argument("--workers", type=int, default=8,
                   help="threads for reading the masks, the loop waits on the "
                        "disk rather than on the processor")
    r.add_argument("--force", action="store_true")
    r.set_defaults(func=run_raster)

    b = sub.add_parser("bericht", help="die Endpunkte und das Tor")
    common(b)
    b.set_defaults(func=run_bericht)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
