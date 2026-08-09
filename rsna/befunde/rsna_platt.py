"""Platt-Kalibrierung und Schwelle fuer Phase 10, aus den Entwicklungsdaten.

Was das Skript tut und warum es das VOR dem Holdout tut
-------------------------------------------------------
Das Modell sortiert gut und rechnet falsch. Auf allen 22872
Entwicklungsbildern, jeweils vorhergesagt von dem Fold-Modell, das dieses Bild
NICHT im Training hatte, sagt es im Mittel 0.334 Krankheitswahrscheinlichkeit
voraus, waehrend 0.225 der Bilder wirklich eine Pneumonie tragen. Der Grund ist
kein Fehler, sondern das Umgewichten der seltenen Klasse im Training: es
verschiebt die Skala.

Die Reparatur ist eine logistische Kurve mit zwei Parametern ueber der
Modellausgabe (Platt 1999). Zwei Parameter koennen kaum ueberanpassen, und die
Kurve ist streng monoton, kann die Reihenfolge innerhalb eines Modells also
nicht aendern. Was sie aendert, ist allein die Beschriftung der Achse.

Gefittet wird je Fold-Modell auf dessen INNEREM Selektionssplit
(`sel_f{k}_s0.csv`, 3050 Bilder). Diese Bilder stecken weder im Training noch
in der Bewertung dieses Modells. Der Bewertungsfold (`rsna_f{k}_s0.csv`) wird
NICHT zum Fitten benutzt, sondern nur, um die Wirkung zu berichten und die
Schwelle zu ziehen; er ist fuer sein eigenes Modell ebenfalls ungesehen.

Der Holdout kommt hier NICHT vor. Kurve und Schwelle stehen fest, bevor er
gerechnet wird. Wer zuerst rechnet und danach die Kurve waehlt, hat den Holdout
benutzt. Siehe `erklaerungen/29_phase10_final.md`, Abschnitt 7 und 9.

Warum je Fold eine eigene Kurve und nicht eine gemeinsame
--------------------------------------------------------
Weil das ausgelieferte Modell ein Ensemble aus fuenf Modellen ist und es fuer
das Ensemble keinen unbefleckten Satz gibt: jedes Entwicklungsbild wurde von
mindestens vier der fuenf Modelle im Training gesehen. Der Ausweg ist, jedes
Mitglied einzeln sauber zu kalibrieren und danach die fuenf
WAHRSCHEINLICHKEITEN zu mitteln. Der Mittelwert kalibrierter
Wahrscheinlichkeiten ist selbst annaehernd kalibriert, tendenziell eine Spur zu
vorsichtig. Wie gut das aufgeht, misst der Holdout und nicht dieses Skript.

Die Ausgabe lesen
-----------------
Je Fold stehen Steigung und Achsenabschnitt der Kurve. Eine Steigung unter 1
heisst: das Modell war zu selbstsicher, die Kurve zieht die Werte zur Mitte.
Danach kommt die Wirkung auf den Entwicklungsdaten. Der ECE ist der
Kalibrierfehler: alle Bilder nach vorhergesagter Wahrscheinlichkeit in zehn
Faecher sortieren und je Fach messen, wie weit Vorhersage und Wirklichkeit
auseinanderliegen. Der Brier-Wert ist der mittlere quadratische Fehler auf der
Wahrscheinlichkeitsskala. Beide sollen fallen. Die geschichtete AUC soll sich
NICHT nennenswert bewegen; tut sie es doch um mehr als 0.01, stimmt etwas
nicht, denn eine monotone Kurve kann sie je Modell gar nicht aendern.

Zum Schluss die Schwelle am Youden-Punkt, gesamt und je Aufnahmeart. Die Luecke
zwischen AP und PA ist kein Fehler des Fits, sondern die bekannte Folge einer
gemeinsamen Schwelle bei ungleicher Haeufigkeit. Sie gehoert in die App und ins
README, nicht wegerklaert.

CLI:
  python rsna/befunde/rsna_platt.py
  python rsna/befunde/rsna_platt.py --pred-dir predictions_final_model --out serving/model/kalibrierung_p10.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _repo_path  # noqa: F401  (legt die Nachbarordner auf den Importpfad)

# --------------------------------------------------------------------------
# Die Vorfestlegung, als Konstanten. Sie stehen in
# erklaerungen/29_phase10_final.md und werden von tests/test_rsna_phase10.py
# dagegen gelesen.
# --------------------------------------------------------------------------
ARM_TAG = "_p5head_ex"
ARM_DIR = "predictions_final_model"
FOLDS = [0, 1, 2, 3, 4]
CKPT = [f"checkpoints/rsna_f{k}_s0_p5head_ex.pth" for k in FOLDS]

SOLL_DEV_N = 22872          # Entwicklungsbilder, ohne Holdout
SOLL_SEL_N = 3050           # innerer Selektionssplit je Fold
AUC_DRIFT_MAX = 0.01        # so viel darf die Kalibrierung A hoechstens bewegen

EPS = 1e-6


def abbruch(text: str) -> None:
    print(f"\nABBRUCH: {text}")
    sys.exit(1)


# --------------------------------------------------------------------------
# Statistik, hier geschrieben und nicht importiert, damit eine zweite
# Umsetzung existiert. Dieselbe Regel wie in den Auswertungen der Phasen 6
# bis 9.
# --------------------------------------------------------------------------

def rank_auc(score, label) -> float:
    """AUC als Rangstatistik. Bindungen bekommen ihren mittleren Rang."""
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    if label.all() or not label.any():
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    n1 = int(label.sum())
    n0 = int((~label).sum())
    return float((r[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def stratified_auc(score, label, view) -> float:
    """Der n-gewichtete Mittelwert der AUC in AP und der AUC in PA."""
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
    """Mittlerer Kalibrierfehler ueber gleich breite Faecher."""
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


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def platt_fit(p: np.ndarray, y: np.ndarray,
              schritte: int = 100, tol: float = 1e-10) -> tuple[float, float]:
    """Unbestrafte logistische Regression auf EINEM Merkmal, per Newton.

    Absichtlich von Hand und nicht mit sklearn: `LogisticRegression` ist
    voreingestellt BESTRAFT (C=1.0), und eine Bestrafung auf zwei Parametern
    zieht die Kurve unbemerkt zur Diagonalen zurueck. Wer sie abschalten will,
    muss C sehr gross setzen und weiss dann immer noch nicht, ob der Loeser
    frueh abgebrochen hat. Newton auf zwei Parametern ist zwanzig Zeilen und
    konvergiert hier in weniger als zehn Schritten.
    """
    z = logit(p)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([z, np.ones_like(z)])
    w = np.zeros(2)
    for _ in range(schritte):
        eta = X @ w
        mu = 1.0 / (1.0 + np.exp(-eta))
        s = np.clip(mu * (1.0 - mu), 1e-12, None)
        grad = X.T @ (y - mu)
        H = (X * s[:, None]).T @ X
        schritt = np.linalg.solve(H, grad)
        w = w + schritt
        if np.max(np.abs(schritt)) < tol:
            break
    else:
        abbruch("die Platt-Kurve ist nicht konvergiert")
    return float(w[0]), float(w[1])


def platt_apply(p, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * logit(p) + b)))


def youden(y, p) -> tuple[float, float, float]:
    """Schwelle mit dem groessten Abstand Sensitivitaet plus Spezifitaet minus 1.

    Ueber die vorkommenden Werte selbst, nicht ueber ein Raster, damit die
    Schwelle exakt auf einem Datenpunkt sitzt.
    """
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p, dtype=float)
    ordnung = np.argsort(-p)
    ys = y[ordnung]
    ps = p[ordnung]
    n1 = int(ys.sum())
    n0 = int((~ys).sum())
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    j = tp / n1 - fp / n0
    # nur an Stellen werten, an denen der naechste Wert kleiner ist
    gueltig = np.r_[ps[1:] < ps[:-1], True]
    j = np.where(gueltig, j, -np.inf)
    i = int(np.argmax(j))
    return float(ps[i]), float(tp[i] / n1), float(1.0 - fp[i] / n0)


def kennzahlen_bei(y, p, thr: float) -> tuple[float, float]:
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(p, dtype=float) >= thr
    sens = float(pred[y].mean()) if y.any() else float("nan")
    spez = float((~pred[~y]).mean()) if (~y).any() else float("nan")
    return sens, spez


def pruefsumme(pfad: Path) -> str:
    h = hashlib.sha256()
    h.update(pfad.read_bytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", type=Path, default=Path(ARM_DIR))
    p.add_argument("--out", type=Path,
                   default=Path("serving") / "model" / "kalibrierung_p10.json")
    p.add_argument("--kein-schreiben", action="store_true",
                   help="nur rechnen und drucken, nichts ablegen")
    args = p.parse_args()

    print("=" * 78)
    print("PLATT-KALIBRIERUNG UND SCHWELLE, nur Entwicklungsdaten")
    print("=" * 78)
    print(f"  Arm {ARM_TAG} aus {args.pred_dir}")
    print("  Der Holdout kommt in diesem Skript nicht vor.")

    if not args.pred_dir.is_dir():
        abbruch(f"{args.pred_dir} gibt es nicht")

    kurven, teile, quellen = [], [], {}
    print()
    print("  Fold   n(sel)   Steigung   Achsenabschnitt   A(val) roh   A(val) kal")
    for k in FOLDS:
        f_sel = args.pred_dir / f"sel_f{k}_s0.csv"
        f_val = args.pred_dir / f"rsna_f{k}_s0.csv"
        for f in (f_sel, f_val):
            if not f.is_file():
                abbruch(f"{f} fehlt")
            quellen[f.name] = pruefsumme(f)

        sel = pd.read_csv(f_sel)
        val = pd.read_csv(f_val)
        if len(sel) != SOLL_SEL_N:
            abbruch(f"sel_f{k} hat {len(sel)} Zeilen, erwartet {SOLL_SEL_N}")
        for spalte in ("y", "viewpos", "p_sel"):
            if spalte not in sel.columns:
                abbruch(f"sel_f{k} hat keine Spalte '{spalte}'")
        for spalte in ("patientId", "y", "viewpos", "p_clean"):
            if spalte not in val.columns:
                abbruch(f"rsna_f{k} hat keine Spalte '{spalte}'")

        a, b = platt_fit(sel.p_sel.values, sel.y.values)
        # Monotonie ist keine Annahme, sondern wird geprueft: eine negative
        # Steigung wuerde die Reihenfolge umdrehen und aus der Kalibrierung
        # eine Verschlechterung machen.
        if a <= 0:
            abbruch(f"Fold {k}: die Steigung ist {a:.4f} und damit nicht monoton")

        pk = platt_apply(val.p_clean.values, a, b)
        a_roh = stratified_auc(val.p_clean.values, val.y.values, val.viewpos.values)
        a_kal = stratified_auc(pk, val.y.values, val.viewpos.values)
        if abs(a_kal - a_roh) > AUC_DRIFT_MAX:
            abbruch(f"Fold {k}: die Kalibrierung bewegt A um {a_kal - a_roh:+.4f}, "
                    f"erlaubt sind {AUC_DRIFT_MAX}. Eine monotone Kurve kann das "
                    f"nicht; die Verkabelung stimmt nicht.")

        kurven.append({"fold": k, "a": a, "b": b})
        d = val[["patientId", "y", "viewpos", "p_clean"]].copy()
        d["p_kal"] = pk
        d["fold"] = k
        teile.append(d)
        print(f"  {k:>4}   {len(sel):>6}   {a:>8.4f}   {b:>15.4f}   "
              f"{a_roh:>10.4f}   {a_kal:>10.4f}")

    oof = pd.concat(teile, ignore_index=True)
    if len(oof) != SOLL_DEV_N:
        abbruch(f"{len(oof)} Vorhersagen zusammengelegt, erwartet {SOLL_DEV_N}")
    if oof.patientId.duplicated().any():
        abbruch("ein Bild kommt in zwei Bewertungsfolds vor, die Teilung ist kaputt")

    y = oof.y.values
    print()
    print("  Jedes der 22872 Bilder ist genau einmal dabei, vorhergesagt von")
    print("  dem Modell, das es NICHT im Training hatte.")

    werte = {
        "n": int(len(oof)),
        "praevalenz": float(y.mean()),
        "mittel_roh": float(oof.p_clean.mean()),
        "mittel_kal": float(oof.p_kal.mean()),
        "brier_roh": brier(y, oof.p_clean.values),
        "brier_kal": brier(y, oof.p_kal.values),
        "ece_roh": ece(y, oof.p_clean.values),
        "ece_kal": ece(y, oof.p_kal.values),
        "a_roh": stratified_auc(oof.p_clean.values, y, oof.viewpos.values),
        "a_kal": stratified_auc(oof.p_kal.values, y, oof.viewpos.values),
    }

    print()
    print("=" * 78)
    print("WAS DIE KALIBRIERUNG BRINGT, auf den Entwicklungsdaten")
    print("=" * 78)
    print(f"  {'':<40}{'roh':>10}{'kalibriert':>14}")
    print(f"  {'mittlere Vorhersage':<40}{werte['mittel_roh']:>10.4f}"
          f"{werte['mittel_kal']:>14.4f}")
    print(f"  {'  beobachtet':<40}{werte['praevalenz']:>10.4f}"
          f"{werte['praevalenz']:>14.4f}")
    print(f"  {'Brier, kleiner ist besser':<40}{werte['brier_roh']:>10.4f}"
          f"{werte['brier_kal']:>14.4f}")
    print(f"  {'ECE, mittlerer Kalibrierfehler':<40}{werte['ece_roh']:>10.4f}"
          f"{werte['ece_kal']:>14.4f}")
    print(f"  {'A, geschichtete AUC':<40}{werte['a_roh']:>10.4f}"
          f"{werte['a_kal']:>14.4f}")
    print()
    print("  A soll sich NICHT bewegen. Dass es leicht steigt, liegt daran, dass")
    print("  die fuenf Modelle nach der Kalibrierung auf einer gemeinsamen Skala")
    print("  liegen und im zusammengelegten Vergleich nicht mehr gegeneinander")
    print("  verrutschen. Je Fold einzeln ist die Aenderung null bis auf")
    print("  Rundung, siehe die Tabelle oben.")

    thr, sens, spez = youden(y, oof.p_kal.values)
    print()
    print("=" * 78)
    print("DIE SCHWELLE, am Youden-Punkt der kalibrierten Vorhersagen")
    print("=" * 78)
    print(f"  {'':<22}{'Schwelle':>10}{'Sens':>10}{'Spez':>10}{'n':>8}{'Praev':>8}")
    print(f"  {'gemeinsam':<22}{thr:>10.4f}{sens:>10.4f}{spez:>10.4f}"
          f"{len(oof):>8}{y.mean():>8.3f}")
    je_view = {}
    for v in ("AP", "PA"):
        m = oof.viewpos.values == v
        s, sp = kennzahlen_bei(y[m], oof.p_kal.values[m], thr)
        je_view[v] = {"n": int(m.sum()), "praevalenz": float(y[m].mean()),
                      "sens": s, "spez": sp}
        print(f"  {'davon ' + v:<22}{'':>10}{s:>10.4f}{sp:>10.4f}"
              f"{int(m.sum()):>8}{y[m].mean():>8.3f}")
    luecke = abs(je_view["AP"]["sens"] - je_view["PA"]["sens"])
    print(f"  Sensitivitaetsluecke {luecke:.4f}")
    print()
    print("  Die Luecke ist kein Fehler des Fits. Eine gemeinsame Schwelle ist")
    print("  bei ungleicher Haeufigkeit in den beiden Aufnahmearten praktisch")
    print("  ein anderer Test. Getrennte Schwellen waeren die Loesung und sind")
    print("  nicht ausrollbar: die App kennt die Projektion nicht.")

    nutzlast = {
        "erzeugt_von": "rsna/befunde/rsna_platt.py",
        "vorfestlegung": "erklaerungen/29_phase10_final.md",
        "arm": ARM_TAG,
        "pred_dir": str(args.pred_dir).replace("\\", "/"),
        "checkpoints": CKPT,
        "platt": kurven,
        "schwelle": thr,
        "schwelle_herkunft": ("Youden auf den kalibrierten Vorhersagen-ohne-Training, "
                              f"{len(oof)} Entwicklungsbilder, gemeinsam ueber beide "
                              "Aufnahmearten"),
        "dev": werte,
        "dev_bei_schwelle": {"sens": sens, "spez": spez, "je_view": je_view,
                             "sens_luecke": luecke},
        "quellen": quellen,
    }

    if args.kein_schreiben:
        print("\n  --kein-schreiben gesetzt, nichts abgelegt.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(nutzlast, indent=2), encoding="utf-8")
    print()
    print(f"  geschrieben: {args.out}")
    print("  Diese Datei ist ab jetzt die Wahrheit fuer die App UND fuer die")
    print("  Holdout-Auswertung. Wer sie nach dem Holdout aendert, hat den")
    print("  Holdout benutzt.")


if __name__ == "__main__":
    main()
