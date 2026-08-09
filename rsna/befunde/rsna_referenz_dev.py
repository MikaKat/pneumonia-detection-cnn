"""Die Referenzverteilung der Entwicklungsdaten fuer die Einordnung in der App.

Wozu
----
Die App zeigt eine kalibrierte Wahrscheinlichkeit auf einem Balken von 0 bis
100 Prozent. Ohne Bezugsgroesse liest man sie gegen die gefuehlte Mitte bei
50 Prozent, und die ist hier falsch: der Arbeitspunkt liegt bei 0,2003, der
hoechste je gemessene Wert bei 0,8927. Diese Datei liefert den Satz
"hoeher als X Prozent der Entwicklungsbilder".

Woraus
------
Aus denselben 22872 Out-of-fold-Vorhersagen, aus denen `rsna_platt.py` Kurven
und Schwelle gezogen hat: je Bild die Vorhersage des Modells, das dieses Bild
NICHT im Training hatte, danach die Platt-Kurve seines Folds. Kein neuer
Durchlauf, keine neuen Gewichte.

**Der Holdout wird hier NICHT angefasst.** Er ist als Messgroesse verbraucht;
eine Zahl aus ihm dauerhaft in der laufenden App anzuzeigen hiesse, ihn zur
staendigen Referenz zu machen.

Der bekannte Unterschied, und warum er getragen wird
----------------------------------------------------
Auf den Entwicklungsdaten liegt je Bild EINE Vorhersage vor, die App zeigt das
Mittel aus fuenf Modellen, und Mitteln zieht die Extreme ein. Die beiden
Verteilungen sind also nicht dieselbe Groesse. Wie stark sie auseinandergehen,
ist auf dem Holdout an den bereits gerechneten Spalten gemessen worden
(`erklaerungen/31_webapp_karten_und_skala.md`, Abschnitt 3): hoechstens
1,7 Prozentpunkte ueber den ganzen Bereich. Dieses Skript rechnet den Vergleich
nach, wenn die Holdout-Datei da ist, und BRICHT AB, wenn er groesser wird als
die hier festgeschriebene Toleranz. So bleibt die Aussage der App an eine
gepruefte Groesse gebunden statt an eine einmal notierte Zahl.

Ausgabe
-------
`serving/model/referenz_dev.json`, ein Quantilraster mit 1001 Stuetzstellen
(0,0 bis 100,0 Prozent in Schritten von 0,1). Rund 15 kB statt 22872
Einzelwerten, und fein genug: eine Stuetzstelle deckt rund 23 Bilder ab.

Aufruf:  python rsna/befunde/rsna_referenz_dev.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _repo_path  # noqa: F401

WURZEL = Path(__file__).resolve().parents[2]
KALIBRIERUNG = WURZEL / "serving" / "model" / "kalibrierung_p10.json"
HOLDOUT = WURZEL / "predictions_holdout" / "holdout.csv"
ZIEL = WURZEL / "serving" / "model" / "referenz_dev.json"

SOLL_DEV_N = 22872          # wie in rsna_platt.py
STUETZSTELLEN = 1001        # 0,0 bis 100,0 Prozent in Schritten von 0,1
TOLERANZ_PP = 2.5           # erlaubte Abweichung Einzelmodell gegen Ensemble


def abbruch(text: str) -> None:
    print(f"ABBRUCH: {text}", file=sys.stderr)
    raise SystemExit(1)


def logit(p, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def platt_apply(p, a: float, b: float) -> np.ndarray:
    """Dieselbe Formel und dasselbe eps wie rsna_holdout.py und serving/main.py."""
    return 1.0 / (1.0 + np.exp(-(a * logit(p) + b)))


def perzentil(raster: np.ndarray, x: float) -> float:
    """Anteil der Verteilung unter x, in Prozent, aus dem Quantilraster."""
    i = int(np.searchsorted(raster, x, side="left"))
    return 100.0 * i / (len(raster) - 1)


def main() -> None:
    if not KALIBRIERUNG.is_file():
        abbruch(f"{KALIBRIERUNG} fehlt.")
    kal = json.loads(KALIBRIERUNG.read_text(encoding="utf-8"))
    pred_dir = WURZEL / kal["pred_dir"]
    kurven = {int(e["fold"]): (float(e["a"]), float(e["b"])) for e in kal["platt"]}

    teile = []
    for k, (a, b) in sorted(kurven.items()):
        pfad = pred_dir / f"rsna_f{k}_s0.csv"
        if not pfad.is_file():
            abbruch(f"{pfad} fehlt.")
        val = pd.read_csv(pfad)
        for spalte in ("patientId", "y", "p_clean"):
            if spalte not in val.columns:
                abbruch(f"Spalte {spalte!r} fehlt in {pfad.name}.")
        teile.append(pd.DataFrame({
            "patientId": val.patientId.values,
            "y": val.y.values,
            "p_kal": platt_apply(val.p_clean.values, a, b),
        }))
        print(f"  Fold {k}: {len(val):>6} Bilder, Platt a={a:.4f} b={b:.4f}")

    oof = pd.concat(teile, ignore_index=True)
    if len(oof) != SOLL_DEV_N:
        abbruch(f"{len(oof)} Out-of-fold-Zeilen, erwartet {SOLL_DEV_N}.")
    if oof.patientId.duplicated().any():
        abbruch("mindestens ein Bild kommt mehrfach vor; die Folds ueberlappen.")

    # Gegenprobe gegen die Kalibrierdatei: derselbe Rechenweg muss denselben
    # Mittelwert liefern. Ohne sie wuerde eine vertauschte Kurve hier eine
    # Verteilung erzeugen, die von aussen wie eine aussieht.
    soll = float(kal["dev"]["mittel_kal"])
    ist = float(oof.p_kal.mean())
    if abs(ist - soll) > 1e-6:
        abbruch(f"mittlere kalibrierte Wahrscheinlichkeit {ist:.8f}, "
                f"in der Kalibrierdatei steht {soll:.8f}.")
    print(f"\n{len(oof)} Bilder, Mittel {ist:.6f} (Kalibrierdatei {soll:.6f}) OK")

    werte = np.sort(oof.p_kal.values)
    raster = np.quantile(werte, np.linspace(0.0, 1.0, STUETZSTELLEN))

    # Die Toleranzpruefung. Sie braucht den Holdout, weil nur dort Einzelmodell
    # UND Ensemble fuer dieselben Bilder vorliegen; gelesen werden ausschliess-
    # lich schon gerechnete Spalten, es entsteht keine neue Kennzahl.
    vergleich = None
    if HOLDOUT.is_file():
        hold = pd.read_csv(HOLDOUT)
        ens = np.sort(hold.p_ens.values)
        einzel = np.sort(np.concatenate([hold[f"p_kal_f{k}"].values for k in sorted(kurven)]))
        zeilen, groesste = [], 0.0
        for x in (float(kal["schwelle"]), 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            pe = 100.0 * float((ens < x).mean())
            ps = 100.0 * float((einzel < x).mean())
            zeilen.append({"p": round(x, 4), "perzentil_ensemble": round(pe, 1),
                           "perzentil_einzelmodell": round(ps, 1),
                           "differenz_pp": round(pe - ps, 1)})
            groesste = max(groesste, abs(pe - ps))
        print(f"\nEinzelmodell gegen Ensemble, groesste Abweichung "
              f"{groesste:.1f} Prozentpunkte (Toleranz {TOLERANZ_PP})")
        if groesste > TOLERANZ_PP:
            abbruch(f"die Abweichung {groesste:.1f} pp ueberschreitet die "
                    f"Toleranz {TOLERANZ_PP} pp. Der Satz in der Oberflaeche "
                    f"waere dann nicht mehr gedeckt.")
        vergleich = {"toleranz_pp": TOLERANZ_PP,
                     "groesste_abweichung_pp": round(groesste, 1),
                     "tabelle": zeilen}
    else:
        print(f"\nHINWEIS: {HOLDOUT} fehlt, die Toleranzpruefung entfaellt.")

    ZIEL.write_text(json.dumps({
        "erzeugt_von": "rsna/befunde/rsna_referenz_dev.py",
        "erklaerung": "erklaerungen/31_webapp_karten_und_skala.md, Abschnitt 3",
        "arm": kal["arm"],
        "quelle": f"{kal['pred_dir']}/rsna_f{{0..4}}_s0.csv, Platt je Fold",
        "was": "kalibrierte Out-of-fold-Wahrscheinlichkeiten der "
               "Entwicklungsbilder, EIN Modell je Bild",
        "n": int(len(oof)),
        "praevalenz": round(float(oof.y.mean()), 6),
        "mittel": round(ist, 6),
        "schwelle": kal["schwelle"],
        "raster_schritt_prozent": round(100.0 / (STUETZSTELLEN - 1), 4),
        "raster": [round(float(v), 6) for v in raster],
        "einzelmodell_gegen_ensemble": vergleich,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{ZIEL.relative_to(WURZEL)} geschrieben "
          f"({ZIEL.stat().st_size / 1024:.1f} kB)")
    for x in (0.05, float(kal["schwelle"]), 0.45, 0.6, 0.8):
        print(f"  p = {x:.4f}  ->  hoeher als {perzentil(raster, x):.1f} % der Entwicklungsbilder")


if __name__ == "__main__":
    main()
