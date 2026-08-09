"""Das Urteil der Phase 10, aus den Holdout-Vorhersagen.

Die Regeln stehen in `erklaerungen/29_phase10_final.md`, Abschnitt 9, und sind
hier als Konstanten verdrahtet. `tests/test_rsna_phase10.py` liest jede einzelne
gegen das Dokument nach, damit ein stilles Verschieben im Diff auffaellt.

Der primaere Endpunkt
---------------------
A auf dem Holdout, das Ensemble aus fuenf kalibrierten Modellen. Geschichtete
AUC, also erst innerhalb der AP-Bilder gerechnet, dann innerhalb der PA-Bilder,
dann n-gewichtet gemittelt. Ohne diese Trennung bekaeme das Modell Punkte
dafuer, dass es die Aufnahmeart erkennt, und genau das ist das Problem dieses
Projekts und nicht seine Loesung.

Das Tor: das untere Ende des 90-Prozent-Intervalls liegt ueber 0.80. Die Latte
sind drei Foldstreuungen unter der Kreuzvalidierungsschaetzung (0.8368 minus
3 mal 0.0105 ergibt 0.8052, abgerundet auf 0.80). Ein Abfall darunter waere kein
Rauschen mehr, sondern hiesse, dass die Entwicklung sich an ihre eigenen Daten
angepasst hat.

Das Intervall kommt aus einem geschichteten Bootstrap ueber Bilder: es wird
innerhalb jeder Zelle aus Aufnahmeart mal Diagnose mit Zuruecklegen gezogen, so
dass Haeufigkeit und Mischung erhalten bleiben und nur die Auswahl der Patienten
wackelt. 2000 Ziehungen, Perzentile 5 und 95.

Alles Weitere ist sekundaer, vorher benannt und ohne Tor: C, die Kalibrierung,
Sensitivitaet und Spezifitaet an der vorher gezogenen Schwelle, und der
Vergleich der fuenf Einzelmodelle mit ihrem jeweiligen
Kreuzvalidierungswert. Der letzte Punkt ist die Ehrlichkeitsprobe der ganzen
Arbeit: war die Schaetzung aus der Kreuzvalidierung optimistisch?

Was dieses Skript NICHT tut
---------------------------
Es aendert nichts. Faellt das Tor, wird das berichtet und nicht repariert. Die
Schwelle, die Kurven und die Zusammensetzung des Ensembles stehen in
`serving/model/kalibrierung_p10.json` und stammen ausschliesslich aus den
Entwicklungsdaten.

CLI:
  python rsna/befunde/rsna_phase10_auswertung.py
  python rsna/befunde/rsna_phase10_auswertung.py --nur-urteil
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _repo_path  # noqa: F401  (legt die Nachbarordner auf den Importpfad)

# --------------------------------------------------------------------------
# Die Vorfestlegung, als Konstanten
# --------------------------------------------------------------------------
LATTE_A = 0.80              # das Tor: unteres Intervallende darueber
NIVEAU = 0.90               # zweiseitig, Perzentile 5 und 95
ZIEHUNGEN = 2000

ANKER_A = 0.8368            # Kreuzvalidierung, fuenf Folds, Einzelmodell
ANKER_A_SD = 0.0105         # Foldstreuung
VORHER_LO = 0.8122          # 90-Prozent-Vorhersageintervall fuer EINE neue
VORHER_HI = 0.8614          # Messung eines Einzelmodells
ANKER_C = 0.7467

ARM_TAG = "_p5head_ex"
FOLDS = [0, 1, 2, 3, 4]
SOLL_N = 3812
BOOT_SEED = 0


def abbruch(text: str) -> None:
    print(f"\nABBRUCH: {text}")
    sys.exit(1)


# --------------------------------------------------------------------------
# Statistik, hier geschrieben und nicht importiert, damit eine zweite
# Umsetzung existiert
# --------------------------------------------------------------------------

def rank_auc(score, label) -> float:
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    if label.all() or not label.any():
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    n1 = int(label.sum())
    n0 = int((~label).sum())
    return float((r[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def stratified_auc(score, label, view) -> float:
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    view = np.asarray(view)
    teile, gewichte = [], []
    for v in ("AP", "PA"):
        m = view == v
        a = rank_auc(score[m], label[m])
        if np.isfinite(a):
            teile.append(a)
            gewichte.append(int(m.sum()))
    if not teile:
        return float("nan")
    return float(np.average(teile, weights=gewichte))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def ece(y, p, faecher: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    kanten = np.linspace(0.0, 1.0, faecher + 1)[1:-1]
    idx = np.digitize(p, kanten)
    fehler = 0.0
    for b in range(faecher):
        m = idx == b
        if m.any():
            fehler += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(fehler)


def zellen(label, view) -> list[np.ndarray]:
    """Die Zellen des geschichteten Bootstrap: Aufnahmeart mal Diagnose."""
    label = np.asarray(label, dtype=int)
    view = np.asarray(view)
    aus = []
    for v in ("AP", "PA"):
        for c in (0, 1):
            idx = np.flatnonzero((view == v) & (label == c))
            if idx.size:
                aus.append(idx)
    return aus


def boot_intervall(fn, label, view, ziehungen: int = ZIEHUNGEN,
                   niveau: float = NIVEAU, seed: int = BOOT_SEED) -> dict:
    """Perzentilintervall einer Kennzahl, geschichtet ueber Bilder gezogen."""
    rng = np.random.default_rng(seed)
    zs = zellen(label, view)
    werte = np.empty(ziehungen)
    for b in range(ziehungen):
        pick = np.concatenate([rng.choice(z, size=z.size, replace=True) for z in zs])
        werte[b] = fn(pick)
    lo = float(np.percentile(werte, 100 * (1 - niveau) / 2))
    hi = float(np.percentile(werte, 100 * (1 - (1 - niveau) / 2)))
    return {"lo": lo, "hi": hi, "streuung": float(werte.std(ddof=1))}


def kennzahlen_bei(y, p, thr: float) -> tuple[float, float]:
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(p, dtype=float) >= thr
    sens = float(pred[y].mean()) if y.any() else float("nan")
    spez = float((~pred[~y]).mean()) if (~y).any() else float("nan")
    return sens, spez


def urteil_a(punkt: float, lo: float) -> tuple[str, str]:
    """Das Tor: das untere Ende ueber der Latte. Nur das untere Ende zaehlt."""
    if lo > LATTE_A:
        return "BESTANDEN", (f"das untere Ende {lo:.4f} liegt ueber der Latte "
                             f"{LATTE_A:.2f}")
    return "DURCHGEFALLEN", (f"das untere Ende {lo:.4f} liegt nicht ueber der "
                             f"Latte {LATTE_A:.2f}")


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--holdout-dir", type=Path, default=Path("predictions_holdout"))
    p.add_argument("--kalibrierung", type=Path,
                   default=Path("serving") / "model" / "kalibrierung_p10.json")
    p.add_argument("--ergebnisse", type=Path, default=Path("results_rsna.csv"))
    p.add_argument("--nur-urteil", action="store_true")
    args = p.parse_args()

    csv = args.holdout_dir / "holdout.csv"
    sperre = args.holdout_dir / ".holdout_verbraucht.json"
    if not csv.is_file():
        abbruch(f"{csv} fehlt. Erst rsna/pipeline/rsna_holdout.py laufen lassen.")
    if not sperre.is_file():
        abbruch(f"{sperre} fehlt. Die Vorhersagen stammen dann nicht aus dem "
                f"abgesicherten Lauf.")
    lock = json.loads(sperre.read_text(encoding="utf-8"))
    kal = json.loads(args.kalibrierung.read_text(encoding="utf-8"))
    d = pd.read_csv(csv)

    print("=" * 78)
    print("HERKUNFT, vor jeder Zahl")
    print("=" * 78)
    if not lock.get("erster_blick", True):
        print("  ACHTUNG: die Sperrdatei sagt, das war NICHT der erste Blick auf")
        print("  den Holdout. Jede Zahl unten ist entsprechend zu lesen.")
    else:
        print(f"  ok   erster und einziger Blick, gerechnet {lock['wann']}")
    if lock.get("arm") != ARM_TAG or kal.get("arm") != ARM_TAG:
        abbruch(f"Arm stimmt nicht: Lauf {lock.get('arm')!r}, "
                f"Kalibrierung {kal.get('arm')!r}, erwartet {ARM_TAG!r}")
    print(f"  ok   Arm {ARM_TAG}, Hardware {lock.get('device')}")
    if len(d) != SOLL_N:
        abbruch(f"{len(d)} Zeilen, erwartet {SOLL_N}")
    print(f"  ok   {len(d)} Bilder, Praevalenz {d.y.mean():.4f}")
    fehlt = [c for c in ["y", "viewpos", "p_ens"] + [f"p_kal_f{k}" for k in FOLDS]
             if c not in d.columns]
    if fehlt:
        abbruch(f"Spalten fehlen: {fehlt}")

    # Das Ensemble muss wirklich das Mittel der fuenf sein. Ohne diese Pruefung
    # koennte p_ens alles sein, auch ein einzelnes Modell.
    mittel = d[[f"p_kal_f{k}" for k in FOLDS]].mean(axis=1).to_numpy()
    ab = float(np.max(np.abs(mittel - d.p_ens.to_numpy())))
    if ab > 1e-9:
        abbruch(f"p_ens ist nicht das Mittel der fuenf kalibrierten Spalten, "
                f"groesste Abweichung {ab:.2e}")
    print(f"  ok   p_ens ist das Mittel der fuenf   groesste Abweichung {ab:.2e}")

    thr = float(kal["schwelle"])
    if abs(thr - float(lock["schwelle"])) > 1e-12:
        abbruch("die Schwelle der Kalibrierdatei und die des Laufs gehen "
                "auseinander")
    print(f"  ok   Schwelle {thr:.4f}, gezogen VOR dem Holdout "
          f"({kal['schwelle_herkunft']})")

    y = d.y.to_numpy(dtype=float)
    vp = d.viewpos.to_numpy()
    pe = d.p_ens.to_numpy(dtype=float)

    a_ens = stratified_auc(pe, y, vp)
    iv = boot_intervall(lambda i: stratified_auc(pe[i], y[i], vp[i]), y, vp)

    print()
    print("=" * 78)
    print("DAS URTEIL, nach der Vorfestlegung vom 09.08.2026")
    print("=" * 78)
    print("  A misst, ob der Wert die PNEUMONIE vorhersagt, geschichtet nach")
    print("  Aufnahmeart. C misst, ob derselbe Wert die AUFNAHMEART verraet.")
    print()
    print(f"  PRIMAER   A auf dem Holdout, Ensemble aus fuenf")
    print(f"    Punktwert {a_ens:.4f}   {int(NIVEAU * 100)}-Prozent-Intervall "
          f"[{iv['lo']:.4f}, {iv['hi']:.4f}]")
    print(f"    Halbbreite {(iv['hi'] - iv['lo']) / 2:.4f}   "
          f"(geschichteter Bootstrap, {ZIEHUNGEN} Ziehungen)")
    stand, warum = urteil_a(a_ens, iv["lo"])
    print(f"    {stand}: {warum}")

    print()
    print(f"  Zum Vergleich, vor dem Lauf aufgeschrieben:")
    print(f"    Kreuzvalidierung, Einzelmodell   {ANKER_A:.4f} +- {ANKER_A_SD:.4f}")
    print(f"    erwartet fuer EINE neue Messung  [{VORHER_LO:.4f}, {VORHER_HI:.4f}]")
    if a_ens > VORHER_HI:
        print("    Das Ensemble liegt ueber dem Vorhersageintervall des")
        print("    Einzelmodells. Das ist der erwartete Gewinn des Mittelns.")
    elif a_ens < VORHER_LO:
        print("    Das Ensemble liegt UNTER dem Vorhersageintervall des")
        print("    Einzelmodells. Das war nicht erwartet und gehoert erklaert.")

    if args.nur_urteil:
        return

    # ---- sekundaer -------------------------------------------------------
    print()
    print("=" * 78)
    print("SEKUNDAER 1: verraet der Wert weiter die Aufnahmeart?")
    print("=" * 78)
    c_ens = rank_auc(pe, vp == "AP")
    civ = boot_intervall(lambda i: rank_auc(pe[i], vp[i] == "AP"), y, vp)
    print(f"  C auf dem Holdout {c_ens:.4f}   [{civ['lo']:.4f}, {civ['hi']:.4f}]")
    print(f"  C in der Kreuzvalidierung {ANKER_C:.4f}")
    print()
    print("  Kein Tor. Neun Phasen haben C nicht gesenkt; die Zahl steht hier,")
    print("  weil sie ins README gehoert und nicht, weil sie noch etwas")
    print("  entscheidet.")

    print()
    print("=" * 78)
    print("SEKUNDAER 2: ist die angezeigte Zahl eine Wahrscheinlichkeit?")
    print("=" * 78)
    roh = d[[f"p_roh_f{k}" for k in FOLDS]].mean(axis=1).to_numpy()
    print(f"  {'':<40}{'roh':>10}{'kalibriert':>14}")
    print(f"  {'mittlere Vorhersage':<40}{roh.mean():>10.4f}{pe.mean():>14.4f}")
    print(f"  {'  beobachtet':<40}{y.mean():>10.4f}{y.mean():>14.4f}")
    print(f"  {'Brier':<40}{brier(y, roh):>10.4f}{brier(y, pe):>14.4f}")
    print(f"  {'ECE':<40}{ece(y, roh):>10.4f}{ece(y, pe):>14.4f}")
    print()
    print(f"  Auf den Entwicklungsdaten stand der ECE bei "
          f"{kal['dev']['ece_kal']:.4f}. Er wird hier")
    print("  hoeher liegen, denn dort stammten die Kurven aus demselben Satz.")
    print("  Vorher aufgeschrieben: unter 0.03 waere gut, ueber 0.05 ein Befund.")

    print()
    print("=" * 78)
    print("SEKUNDAER 3: die Schwelle in der Praxis")
    print("=" * 78)
    sens, spez = kennzahlen_bei(y, pe, thr)
    print(f"  {'':<22}{'Sens':>10}{'Spez':>10}{'n':>8}{'Praev':>8}")
    print(f"  {'gemeinsam':<22}{sens:>10.4f}{spez:>10.4f}{len(d):>8}{y.mean():>8.3f}")
    je = {}
    for v in ("AP", "PA"):
        m = vp == v
        s, sp = kennzahlen_bei(y[m], pe[m], thr)
        je[v] = s
        print(f"  {'davon ' + v:<22}{s:>10.4f}{sp:>10.4f}"
              f"{int(m.sum()):>8}{y[m].mean():>8.3f}")
    print(f"  Sensitivitaetsluecke {abs(je['AP'] - je['PA']):.4f}   "
          f"(Entwicklungsdaten {kal['dev_bei_schwelle']['sens_luecke']:.4f})")
    print()
    print("  Eine feste Schwelle ist bei ungleicher Haeufigkeit in den beiden")
    print("  Aufnahmearten praktisch ein anderer Test. Getrennte Schwellen sind")
    print("  nicht ausrollbar, weil die App die Projektion nicht kennt.")

    print()
    print("=" * 78)
    print("SEKUNDAER 4: war die Kreuzvalidierung optimistisch?")
    print("=" * 78)
    cv = {}
    if args.ergebnisse.is_file():
        r = pd.read_csv(args.ergebnisse)
        r = r[r.tag == ARM_TAG].drop_duplicates(subset="fold", keep="last")
        cv = {int(row.fold): float(row.auc_stratified) for row in r.itertuples()}
    print(f"  {'Fold':>6}{'A in der CV':>14}{'A auf dem Holdout':>20}"
          f"{'Differenz':>12}")
    diffs = []
    for k in FOLDS:
        ah = stratified_auc(d[f"p_kal_f{k}"].to_numpy(), y, vp)
        if k in cv:
            diffs.append(ah - cv[k])
            print(f"  {k:>6}{cv[k]:>14.4f}{ah:>20.4f}{ah - cv[k]:>+12.4f}")
        else:
            print(f"  {k:>6}{'?':>14}{ah:>20.4f}{'':>12}")
    if diffs:
        m = float(np.mean(diffs))
        print(f"  {'Mittel':>6}{np.mean([cv[k] for k in FOLDS if k in cv]):>14.4f}"
              f"{np.mean([stratified_auc(d[f'p_kal_f{k}'].to_numpy(), y, vp) for k in FOLDS]):>20.4f}"
              f"{m:>+12.4f}")
        print()
        if m < -0.02:
            print("  Die Kreuzvalidierung war optimistisch. Der Abstand gehoert")
            print("  ins README, und zwar mit dieser Zahl.")
        else:
            print("  Kein nennenswerter Abstand. Die Kreuzvalidierungsschaetzung")
            print("  hat gehalten, und das ist die eigentliche Nachricht dieses")
            print("  Abschnitts: neun Phasen lang wurde an einer Zahl gemessen,")
            print("  die auf ungesehenen Daten wieder herauskommt.")

    print()
    print("=" * 78)
    print("WAS DER SEKUNDAERE TEIL NICHT DARF")
    print("=" * 78)
    print("  Er darf das Urteil oben nicht umdeuten, und er darf nichts")
    print("  aendern. Weder Schwelle noch Kurve noch Ensemble werden nach")
    print("  diesem Lauf angefasst.")


if __name__ == "__main__":
    main()
