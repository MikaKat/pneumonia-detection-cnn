"""Die Stufentabelle: was jede der drei Stufen auf jedem Datensatz gekostet hat.

Warum es dieses Skript gibt
---------------------------
Die Tabelle in `webapp/src/stufen.js` und die Datei `qc/stufen_p10.json` wurden
am 13.08.2026 von Hand gerechnet und nur als Ergebnis abgelegt. Das ist genau
die Bauart, gegen die dieses Projekt sonst arbeitet: eine Zahl ohne Erzeuger
kann niemand nachrechnen, und tatsaechlich sind die beiden Dateien
auseinandergelaufen (die JSON trug VinDr in Lesart A, die JS in Lesart M). Hier
steht die Rechnung, einmal, und beide Dateien kommen aus ihr.

Was gerechnet wird
------------------
Drei Stufen, zwei Schwellen:

    p <  t_low                unauffaellig   (Ausschluss)
    t_low <= p < t_high       unklar         (keine Aussage)
    p >= t_high               auffaellig     (Einschluss)

Beide Schwellen kommen AUSSCHLIESSLICH aus den 22872 Entwicklungsbildern, jedes
vorhergesagt von dem Fold-Modell, das es nicht im Training hatte, danach mit der
Platt-Kurve seines Folds kalibriert. Es sind Quantile auf dieser Skala:

    t_low  = 10-Prozent-Quantil der POSITIVEN    -> Sensitivitaet 90 %
    t_high = q-Quantil der NEGATIVEN             -> Spezifitaet q

Der Holdout kommt nicht vor. Kermany und VinDr kommen nicht vor. Beide werden
nur GEMESSEN, nachdem die Schwellen feststehen, und genau das macht ihre Zeilen
in der Tabelle unbefangen.

Die Zielspezifitaet ist ein Parameter
-------------------------------------
Sie stand bis zum 13.08.2026 auf 95 % und steht seitdem auf 90 %. Das ist eine
Aenderung NACH Kenntnis der Verteilung und ist als solche in
`erklaerungen/46_obere_schwelle.md` protokolliert. Der Schalter heisst
`--spez-ziel`, damit die alte Fassung eine Kommandozeile weit entfernt bleibt
und nicht neu ausgegraben werden muss.

Die Quantilmethode ist "higher" und nicht der Standard
------------------------------------------------------
`method="higher"` nimmt den naechsten Wert, der in den Daten wirklich vorkommt,
statt zwischen zwei Nachbarn zu interpolieren. Eine interpolierte Schwelle
liegt zwischen zwei Bildern und gehoert keinem; die Zusage "90 % Sensitivitaet"
gilt dann nur ungefaehr. Mit "higher" ist die Schwelle ein tatsaechlicher
Modellwert und die Zusage exakt einloesbar. Die drei Fassungen liegen rund
0.0004 auseinander, das ist fuer die Tabelle folgenlos und fuer die
Reproduzierbarkeit nicht.

CLI:
  python rsna/befunde/rsna_stufen_tabelle.py
  python rsna/befunde/rsna_stufen_tabelle.py --spez-ziel 0.95      (die alte Fassung)
  python rsna/befunde/rsna_stufen_tabelle.py --js                  (Block fuer stufen.js)
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
# Die Vorfestlegung, als Konstanten. Sens-Ziel steht seit erklaerungen/41_
# fest und wird hier NICHT zum Schalter gemacht: die untere Schwelle ist die,
# an der ein Fehler teuer wird, und sie bleibt, wo sie war.
SENS_ZIEL = 0.90
SPEZ_ZIEL_STANDARD = 0.90  # bis 13.08.2026: 0.95, siehe erklaerungen/46_
FOLDS = (0, 1, 2, 3, 4)
EPS = 1e-6

# Die drei Datensaetze, in der Reihenfolge, in der sie in der App stehen.
# `heimat` markiert den, auf dem die Schwellen gewaehlt wurden. Er ist keine
# externe Bestaetigung und darf nie als eine gelesen werden.
QUELLEN = [
    dict(id="rsna", name="RSNA development set",
         was="adults and children, US emergency departments", heimat=True),
    dict(id="vindr", name="VinDr-CXR",
         was="adults, Vietnam, two of three radiologists", heimat=False,
         datei="predictions_extern_vindr/extern_vindr_ens_M.csv",
         spalte_p="p_ens", spalte_y="y"),
    dict(id="kermany", name="Kermany",
         was="children aged one to five, Guangzhou", heimat=False,
         datei="predictions_extern_kermany/extern_kermany_ens.csv",
         spalte_p="p_stretch_ens", spalte_y="label"),
]


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def platt_apply(p, a: float, b: float) -> np.ndarray:
    """Dieselbe Formel und dasselbe eps wie in rsna_platt.py und in main.py.

    Wer eine der drei aendert, aendert die anderen mit, sonst zeigt die App
    eine andere Zahl an als die, mit der die Schwelle gewaehlt wurde.
    """
    return 1.0 / (1.0 + np.exp(-(a * logit(p) + b)))


def entwicklungsvorhersagen(wurzel: Path, kal: dict) -> tuple[np.ndarray, np.ndarray]:
    """Die 22872 Out-of-fold-Vorhersagen, kalibriert, als (p, y)."""
    kurven = {d["fold"]: (d["a"], d["b"]) for d in kal["platt"]}
    pred = wurzel / kal["pred_dir"]
    ps, ys = [], []
    for k in FOLDS:
        d = pd.read_csv(pred / f"rsna_f{k}_s0.csv")
        a, b = kurven[k]
        ps.append(platt_apply(d["p_clean"].to_numpy(), a, b))
        ys.append(d["y"].to_numpy())
    p = np.concatenate(ps)
    y = np.concatenate(ys).astype(bool)
    # Gegentest gegen die Kalibrierdatei. Faellt hier etwas auseinander, ist
    # entweder eine Vorhersagedatei ausgetauscht worden oder die Kurve, und
    # dann waeren alle Zahlen unten still falsch.
    soll = kal["dev"]
    for name, ist, sollwert in [("n", len(p), soll["n"]),
                                ("praevalenz", y.mean(), soll["praevalenz"]),
                                ("mittel_kal", p.mean(), soll["mittel_kal"])]:
        if abs(float(ist) - float(sollwert)) > 1e-9:
            raise SystemExit(
                f"ABBRUCH: {name} ist {ist!r}, die Kalibrierdatei sagt {sollwert!r}"
            )
    return p, y


def stufen(p: np.ndarray, y: np.ndarray, t_low: float, t_high: float) -> dict:
    lo = p < t_low
    hi = p >= t_high
    mid = ~lo & ~hi
    out = {"n": int(len(p)), "praevalenz": float(y.mean())}
    for schluessel, maske in [("low", lo), ("mid", mid), ("high", hi)]:
        out[schluessel] = {
            "anteil": float(maske.mean()),
            "krank": float(y[maske].mean()) if maske.any() else float("nan"),
            "n": int(maske.sum()),
            "n_krank": int(y[maske].sum()),
        }
    # Sensitivitaet der Regel "nicht unauffaellig". Das ist die Zusage, die an
    # der unteren Schwelle haengt, und sie ist unabhaengig von t_high.
    out["sens"] = float((p >= t_low)[y].mean())
    out["spez_bei_t_high"] = float((p < t_high)[~y].mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wurzel", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--kalibrierung", default="serving/model/kalibrierung_p10.json")
    ap.add_argument("--out", default="qc/stufen_p10.json")
    ap.add_argument("--spez-ziel", type=float, default=SPEZ_ZIEL_STANDARD)
    ap.add_argument("--js", action="store_true",
                    help="den GEMESSEN-Block fuer webapp/src/stufen.js ausgeben")
    a = ap.parse_args()

    kal = json.loads((a.wurzel / a.kalibrierung).read_text(encoding="utf-8"))
    p_dev, y_dev = entwicklungsvorhersagen(a.wurzel, kal)

    t_low = float(np.quantile(p_dev[y_dev], 1.0 - SENS_ZIEL, method="higher"))
    t_high = float(np.quantile(p_dev[~y_dev], a.spez_ziel, method="higher"))

    print(f"t_low  {t_low!r}   (Sensitivitaet {SENS_ZIEL:.0%} auf {len(p_dev)} Bildern)")
    print(f"t_high {t_high!r}   (Spezifitaet {a.spez_ziel:.0%} auf denselben Bildern)")
    print()

    ergebnis = {"t_low": t_low, "t_high": t_high,
                "quelle": f"{len(p_dev)} Entwicklungsbilder, out of fold",
                "ziele": {"t_low": f"Sensitivitaet {SENS_ZIEL:.0%}",
                          "t_high": f"Spezifitaet {a.spez_ziel:.0%}",
                          "festgelegt": "erklaerungen/41_stufen_statt_verdikt.md, "
                                        "obere Schwelle geaendert in "
                                        "erklaerungen/46_obere_schwelle.md"},
                "datensaetze": {}}

    zeilen = []
    for q in QUELLEN:
        if q["id"] == "rsna":
            p, y = p_dev, y_dev
        else:
            d = pd.read_csv(a.wurzel / q["datei"])
            p = d[q["spalte_p"]].to_numpy()
            y = d[q["spalte_y"]].to_numpy().astype(bool)
        s = stufen(p, y, t_low, t_high)
        s["name"] = q["name"]
        s["was"] = q["was"]
        s["heimat"] = q["heimat"]
        s["quelle"] = q.get("datei", kal["pred_dir"] + "/rsna_f*_s0.csv + Platt")
        ergebnis["datensaetze"][q["id"]] = s
        zeilen.append((q, s))
        print(f"{q['name']:<22} n {s['n']:>6}  prev {s['praevalenz']:.4f}  "
              f"| unauff {s['low']['anteil']:.4f} ({s['low']['krank']:.4f} krank) "
              f"| unklar {s['mid']['anteil']:.4f} ({s['mid']['krank']:.4f}) "
              f"| auff {s['high']['anteil']:.4f} ({s['high']['krank']:.4f} krank) "
              f"| sens {s['sens']:.4f}")

    ziel = a.wurzel / a.out
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text(json.dumps(ergebnis, indent=2, ensure_ascii=True) + "\n",
                    encoding="utf-8")
    print(f"\ngeschrieben: {ziel}")

    if a.js:
        print("\n// --- Block fuer webapp/src/stufen.js ---")
        print(f"export const T_LOW = {t_low!r};")
        print(f"export const T_HIGH = {t_high!r};")
        print("export const GEMESSEN = [")
        for q, s in zeilen:
            print(f"  {{")
            print(f"    id: {q['id']!r},")
            print(f"    name: {q['name']!r},")
            print(f"    was: {q['was']!r},")
            print(f"    n: {s['n']},")
            print(f"    praevalenz: {round(s['praevalenz'], 4)},")
            for k in ("low", "mid", "high"):
                print(f"    {k}: {{ anteil: {round(s[k]['anteil'], 4)}, "
                      f"krank: {round(s[k]['krank'], 4)} }},")
            print(f"    sens: {round(s['sens'], 4)},")
            print(f"    heimat: {str(q['heimat']).lower()},")
            print(f"  }},")
        print("];")
    return 0


if __name__ == "__main__":
    sys.exit(main())
