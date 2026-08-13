"""VinDr-CXR: everything that can be decided BEFORE a single pixel is read.

Why this runs first
-------------------
On Kermany the bare image dimensions separated the classes at AUC 0.9150. The
external number there is only readable because that leak was measured first and
the model figure was then reported WITHIN its quintiles. That lesson is not
learned twice, so the same check runs here before any inference, and it costs
nothing: the 512 px release ships the ORIGINAL width and height of every image
in `train.csv`, so the leak can be measured without opening one file.

Three things this settles, all of them belonging in the pre-registration:

  1. THE TARGET LABEL. VinDr has 14 findings plus "No finding". RSNA's target is
     "lung opacity consistent with pneumonia", drawn as boxes. The closest match
     is Lung Opacity, Consolidation and Infiltration. The question that has to
     be answered before the run is what counts as NEGATIVE, and there are two
     defensible answers with very different prevalences.
  2. THE LEAK. Grouped cross-validated GradientBoosting on width, height,
     aspect and pixel count, exactly `header_leak` from `rsna_external_kermany`.
  3. WHAT IS MEASURABLE AT ALL. The 512 px release carries no DICOM headers, so
     there is no ViewPosition. C is therefore NOT measurable from this download
     and that has to be stated rather than quietly dropped.

Every image in VinDr is a separate patient, so bootstrap groups are image ids.

  python3 rsna_vindr_vorpruefung.py --csv data/vinbigdata/vinbigdata/train.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# The opacity family. Fixed here, before any number is seen.
ZIEL = ["Lung Opacity", "Consolidation", "Infiltration"]
KEIN_BEFUND = "No finding"


def label_varianten(d: pd.DataFrame) -> pd.DataFrame:
    """Ein Zeile je Bild, mit den drei denkbaren Lesarten des Ziels."""
    g = d.groupby("image_id")
    hat_ziel = g["class_name"].apply(lambda s: s.isin(ZIEL).any())
    nur_kein = g["class_name"].apply(lambda s: (s == KEIN_BEFUND).all())
    n_befunde = g["class_name"].apply(lambda s: (s != KEIN_BEFUND).nunique())
    masse = g[["width", "height"]].first()

    out = pd.DataFrame({"ziel": hat_ziel.astype(int),
                        "nur_kein_befund": nur_kein.astype(int),
                        "andere_befunde": n_befunde}, index=hat_ziel.index)
    out = out.join(masse)
    out["aspect"] = out.width / out.height
    out["pixels"] = out.width * out.height
    # A: negativ ist alles ohne Verschattungsfamilie. Enthaelt die Mittelklasse
    #    (andere Befunde, keine Verschattung) und entspricht damit RSNA, wo 44 %
    #    der Bilder "auffaellig, aber keine Verschattung" sind.
    out["y_A"] = out.ziel
    # B: negativ ist nur "No finding" von allen Radiologen. Sauberer Kontrast,
    #    aber die Mittelklasse faellt raus und das ist NICHT RSNAs Aufgabe.
    out["y_B"] = np.where(out.ziel == 1, 1, np.where(out.nur_kein_befund == 1, 0, -1))
    # K: Positiv nur, wenn MINDESTENS ZWEI Radiologen die Verschattung sahen.
    #    Bei VinDr sieht in 47 % der Faelle nur einer etwas, das Ziel ist also
    #    verrauscht. Diese Lesart tauscht Umfang gegen Sicherheit.
    zust = (d[d.class_name.isin(ZIEL)].groupby("image_id")["rad_id"]
              .nunique().reindex(out.index).fillna(0))
    out["zustimmung"] = zust.astype(int)
    out["y_K"] = np.where(out.zustimmung >= 2, 1,
                          np.where(out.zustimmung == 0, 0, -1))
    return out


def leak(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0):
    """Wortgleich mit `header_leak` aus rsna_external_kermany.py."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    oof = np.zeros(len(y))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups):
        m = GradientBoostingClassifier(random_state=seed).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof)), oof


def einzel_auc(werte: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, werte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=Path("data/vinbigdata/vinbigdata/train.csv"))
    ap.add_argument("--out", type=Path, default=Path("qc/vindr_vorpruefung.json"))
    a = ap.parse_args()

    d = pd.read_csv(a.csv)
    print("=" * 74)
    print("0. WAS DA IST")
    print("=" * 74)
    print(f"  Zeilen {len(d)}, Bilder {d.image_id.nunique()}, "
          f"Radiologen {d.rad_id.nunique()}")
    if "width" not in d.columns:
        raise SystemExit("Diese CSV hat keine width/height. Nimm die des "
                         "512er-Datensatzes, nicht die amtliche.")

    print("\n=== Klassen ===")
    t = (d.groupby("class_name")
           .agg(kaesten=("image_id", "size"), bilder=("image_id", "nunique"))
           .sort_values("bilder", ascending=False))
    for k, r in t.iterrows():
        stern = "  <- Ziel" if k in ZIEL else ""
        print(f"  {k:<24}{r.kaesten:>8}{r.bilder:>8}{stern}")

    b = label_varianten(d)
    nA, posA = len(b), int(b.y_A.sum())
    mB = b.y_B >= 0
    nB, posB = int(mB.sum()), int((b.y_B == 1).sum())

    print("\n" + "=" * 74)
    print("1. DIE ZWEI LESARTEN DES ZIELS")
    print("=" * 74)
    print(f"\n  A  negativ = alles ohne Verschattungsfamilie (MIT Mittelklasse)")
    print(f"     n {nA}, positiv {posA}, Praevalenz {posA / nA:.4f}")
    print(f"  B  negativ = nur 'No finding' von allen (OHNE Mittelklasse)")
    print(f"     n {nB}, positiv {posB}, Praevalenz {posB / nB:.4f}, "
          f"{nA - nB} Bilder verworfen")
    mK = b.y_K >= 0
    nK, posK = int(mK.sum()), int((b.y_K == 1).sum())
    print(f"  K  positiv nur bei MINDESTENS ZWEI Radiologen (Einer-Meinungen raus)")
    print(f"     n {nK}, positiv {posK}, Praevalenz {posK / nK:.4f}, "
          f"{nA - nK} Bilder verworfen")
    print("\n  A entspricht RSNA: dort sind 44 % der Bilder auffaellig OHNE")
    print("  Verschattung, und genau die muss das Modell als negativ erkennen.")
    print("  B waere der leichtere Test und deshalb der unehrlichere.")

    print("\n" + "=" * 74)
    print("2. DER LECK-TEST  (Kermany zum Vergleich: 0,9150)")
    print("=" * 74)
    ergebnis = {}
    mK = b.y_K >= 0
    for name, sub, y in [("A", b, b.y_A.to_numpy()),
                         ("B", b[mB], b[mB].y_B.to_numpy()),
                         ("K", b[mK], b[mK].y_K.to_numpy())]:
        X = sub[["width", "height", "aspect", "pixels"]].to_numpy(float)
        gruppen = sub.index.to_numpy()
        auc, oof = leak(X, y, gruppen)
        print(f"\n  Lesart {name}:  kombiniert {auc:.4f}")
        for sp in ["width", "height", "aspect", "pixels"]:
            print(f"     {sp:<8} allein {einzel_auc(sub[sp].to_numpy(float), y):.4f}")
        ergebnis[f"leak_{name}"] = auc
        if name == "A":
            b.loc[sub.index, "leak_oof"] = oof

    print("\n" + "=" * 74)
    print("3. DIE KAESTEN  (dafuer wurde dieser Datensatz gewaehlt)")
    print("=" * 74)
    k = d[d.class_name.isin(ZIEL)].dropna(subset=["x_min"]).copy()
    k["flaeche"] = ((k.x_max - k.x_min) * (k.y_max - k.y_min)) / (k.width * k.height)
    je_bild = k.groupby("image_id").size()
    je_rad = k.groupby("image_id")["rad_id"].nunique()
    print(f"\n  Bilder mit Ziel-Kaesten   {k.image_id.nunique()}")
    print(f"  Kaesten insgesamt         {len(k)}")
    print(f"  Kaesten je Bild           Median {je_bild.median():.0f}, "
          f"max {je_bild.max()}")
    print(f"  Radiologen je Bild        Median {je_rad.median():.0f}")
    print(f"  Kastenflaeche am Bild     Median {k.flaeche.median():.4f}, "
          f"5./95. Perzentil {k.flaeche.quantile(.05):.4f} / "
          f"{k.flaeche.quantile(.95):.4f}")
    print(f"\n  Bilder, bei denen NUR EIN Radiologe das Ziel sah: "
          f"{int((je_rad == 1).sum())} von {k.image_id.nunique()}")
    print("  -> die Uneinigkeit ist selbst eine Messgroesse und wird berichtet,")
    print("     nicht durch Mehrheitsentscheid weggeraeumt.")

    print("\n" + "=" * 74)
    print("4. WAS DIESER DOWNLOAD NICHT HERGIBT")
    print("=" * 74)
    fehlt = [s for s in ["ViewPosition", "view", "projection"] if s not in d.columns]
    print(f"\n  Keine Projektionsangabe in der CSV ({', '.join(fehlt)} fehlen).")
    print("  Die 512er-PNG tragen keine DICOM-Kopfzeilen. **C IST MIT DIESEM")
    print("  DOWNLOAD NICHT MESSBAR** und faellt aus der Vorfestlegung heraus.")
    print("  Wer C will, zieht ein paar hundert Original-DICOM einzeln nach.")
    print("\n  Ausserdem: die PNG sind auf 512x512 QUADRAT gezogen, das")
    print("  Seitenverhaeltnis steckt nur noch in der CSV. Die 'pad'-Variante")
    print("  laesst sich daher nur ueber ein Zurueckziehen bauen, mit einem")
    print("  zweiten Abtastschritt. Primaer bleibt 'stretch', das ist ohnehin")
    print("  die ausgelieferte Kette.")

    # ---- Die Liste der Positiven, fuer den Maskenlauf.
    # Punkt-AUC ist nur auf Bildern MIT Kaesten definiert, also braucht der
    # U-Net-Lauf 1588 Bilder und nicht 15000. Spaltenname `patientId`, weil
    # `rsna_make_masks --ids-from` genau den erwartet.
    ids = a.out.parent / "vindr_positiv_ids.csv"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patientId": b.index[b.y_A == 1]}).to_csv(ids, index=False)
    print(f"\n  -> {ids}  ({int((b.y_A == 1).sum())} Bilder fuer den Maskenlauf,")
    print(f"     statt {len(b)}. Das ist der Unterschied zwischen Minuten und Stunden.)")

    ergebnis.update({
        "n_bilder": int(nA), "positiv_A": posA, "praevalenz_A": posA / nA,
        "n_B": nB, "positiv_B": posB, "praevalenz_B": posB / nB,
        "bilder_mit_kaesten": int(k.image_id.nunique()),
        "kaesten": int(len(k)),
        "n_K": nK, "positiv_K": posK, "praevalenz_K": posK / nK,
        "ids_datei": str(ids),
        "c_messbar": False,
    })
    a.out.write_text(json.dumps(ergebnis, indent=2))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
