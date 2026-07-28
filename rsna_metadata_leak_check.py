"""
Confounder-Check fuer RSNA -- LAEUFT VOR DEM ERSTEN TRAINING.

Das ist die eine Lehre aus der Kermany-Phase: dort wurde erst trainiert, dann
Grad-CAM angeschaut, dann gestutzt, und erst nach Wochen kam heraus, dass die
JPEG-Abmessungen allein die Klassen mit AUC 0,915 trennen. Jede Modellzahl davor
war unter Vorbehalt zu lesen. Also diesmal umgekehrt.

Die Frage lautet: Wie weit kommt ein Klassifikator, der KEINEN EINZIGEN PIXEL
sieht -- nur den DICOM-Header? Diese Zahl ist die Untergrenze, die das Bildmodell
schlagen muss, um ueberhaupt Radiologie gelernt zu haben.

Erwartung vorab (damit das Ergebnis nicht nachtraeglich passend erklaert wird):

  * Rows/Columns:  konstant 1024x1024. AUC ~0,50. Der Kermany-Confounder
    existiert hier nicht -- das war einer der Gruende fuer den Wechsel.
    Sollte hier doch etwas stehen, ist die Annahme falsch und der Plan neu.
  * ViewPosition:  AP vs. PA. AP heisst ueberwiegend Liegendaufnahme mit
    mobilem Geraet, also kraenkere Patienten. Erwartung deutlich > 0,50.
    Das ist ein ECHTER, aus NIH ChestX-ray14 geerbter Confounder.
  * PatientAge:    Pneumonie haeuft sich in den Randaltern. Erwartung ~0,55-0,65.
  * kombiniert:    vermutlich 0,60-0,70. Deutlich weniger als Kermanys 0,915,
    aber weit ueber Zufall -- und genau deshalb muss spaeter auf age und
    ViewPosition gematcht bzw. geschichtet werden, nicht auf einen Proxy.

Wichtig: ein hoher Wert bedeutet NICHT, dass der Datensatz unbrauchbar ist.
Alter und Aufnahmeart korrelieren real mit Pneumonie -- ein Radiologe weiss das
auch. Es bedeutet, dass eine rohe AUC diesen Anteil enthaelt und die berichtete
Zahl gematcht sein muss.

CLI:
  python rsna_metadata_leak_check.py \
      --dicom data/rsna/stage_2_train_images \
      --csv   data/rsna \
      --out   qc/rsna
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from rsna_data import CLASSES3, load_labels, scan_headers

NUMERIC = ["age_years", "Rows", "Columns", "pixel_spacing"]
CATEGORICAL = ["ViewPosition", "PatientSex"]


def single_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Richtungsunabhaengiger AUC eines Merkmals; NaNs werden ausgelassen."""
    ok = ~np.isnan(s)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2 or len(np.unique(s[ok])) < 2:
        return float("nan")
    a = roc_auc_score(y[ok], s[ok])
    return max(a, 1 - a)


def categorical_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Positivrate je Auspraegung -- ehrlicher als ein in-sample-Target-Encoding."""
    t = df.groupby(col, dropna=False).agg(n=("label", "size"),
                                          pos_rate=("label", "mean"))
    return t.sort_values("n", ascending=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dicom", type=Path, default=Path("data/rsna/stage_2_train_images"))
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--out", type=Path, default=Path("qc/rsna"))
    p.add_argument("--mode", default="clinical", choices=["clinical", "strict"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--limit", type=int, default=None,
                   help="nur die ersten N DICOMs -- Probelauf, NICHT repraesentativ "
                        "(die Dateireihenfolge ist mit dem Label korreliert, s.u.)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lab = load_labels(args.csv, mode=args.mode)
    hdr = scan_headers(args.dicom, cache=args.out / "dicom_headers.csv",
                       limit=args.limit)
    df = lab.merge(hdr, on="patientId", how="inner")
    if df.empty:
        raise SystemExit("Kein Bild in beiden Quellen -- stimmen die Pfade?")

    n_only_csv = len(lab) - len(df)
    y = df["label"].values
    if args.limit:
        print("\n  ACHTUNG: --limit nimmt die ersten N Dateien in Sortierreihenfolge,")
        print("  und die ist NICHT zufaellig: die Positivrate schwankt ueber die")
        print("  sortierte Dateiliste zwischen 0.136 und 0.373 (gesamt 0.225). Die")
        print("  patientId-UUIDs tragen also Batch-Struktur. Zahlen aus einem")
        print("  --limit-Lauf zeigen nur, dass der Code laeuft -- nicht, was gilt.")
    print(f"\n{len(df)} Bilder mit Label und Header"
          f"{f' ({n_only_csv} nur in der CSV)' if n_only_csv else ''}")
    print(f"Modus '{args.mode}': positiv {int(y.sum())} ({y.mean():.3f}), "
          f"negativ {int((y == 0).sum())}")
    print("\nDreiklassige Verteilung:")
    for c in CLASSES3:
        n = int((df["class3"] == c).sum())
        if n:
            print(f"  {c:<32}{n:>7}  ({n / len(df):.3f})")

    # ---- numerische Einzelmerkmale ----
    print(f"\n{'Merkmal':<16}{'neg (Median)':>14}{'pos (Median)':>14}{'AUC':>8}")
    print("-" * 52)
    aucs = {}
    for f in NUMERIC:
        s = pd.to_numeric(df[f], errors="coerce").values.astype(float)
        aucs[f] = single_auc(y, s)
        med = [np.nan if np.isnan(s[y == v]).all() else np.nanmedian(s[y == v])
               for v in (0, 1)]      # nanmedian warnt bei komplett leeren Spalten
        note = ""
        if np.isnan(aucs[f]) and not np.isnan(med[0]) and med[0] == med[1]:
            note = "  <- konstant, kein Confounder moeglich"
        print(f"{f:<16}{med[0]:>14.3f}{med[1]:>14.3f}{aucs[f]:>8.3f}{note}")

    # ---- kategoriale Merkmale ----
    for c in CATEGORICAL:
        print(f"\n{c}: Positivrate je Auspraegung")
        t = categorical_table(df, c)
        for lvl, row in t.iterrows():
            print(f"  {str(lvl):<12}{int(row['n']):>8}{row['pos_rate']:>10.3f}")
        if len(t) == 2:  # binaer -> direkt als AUC lesbar
            ind = (df[c] == t.index[0]).astype(float).values
            aucs[c] = single_auc(y, ind)
            print(f"  -> als Einzelmerkmal: AUC {aucs[c]:.3f}")

    # Kreuztabelle Ansicht x dreiklassige Klasse: zeigt, ob der ViewPosition-
    # Effekt an der Pneumonie haengt oder an "krank ueberhaupt".
    if df["ViewPosition"].nunique() > 1:
        print("\nViewPosition x Klasse (Zeilenanteile):")
        ct = pd.crosstab(df["ViewPosition"], df["class3"], normalize="index")
        print(ct.round(3).to_string())

    # ---- kombiniert, patientengruppierte CV ----
    X = df[NUMERIC].apply(pd.to_numeric, errors="coerce")
    for c in CATEGORICAL:
        d = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True)
        X = pd.concat([X, d], axis=1)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0).values.astype(float)

    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=0)
    prob = cross_val_predict(GradientBoostingClassifier(random_state=0), X, y,
                             groups=df["group"].values, cv=cv,
                             method="predict_proba")[:, 1]
    auc = roc_auc_score(y, prob)
    df["header_score"] = prob

    print(f"\n>> Nur-Header-Klassifikator, {args.folds}-fold gruppierte CV: "
          f"AUC = {auc:.3f}")
    print("   Kein Pixel gesehen. Diese Zahl muss das Bildmodell schlagen,")
    print("   und die berichtete Modell-AUC muss auf diese Groessen gematcht sein.")
    print(f"   (Kermany zum Vergleich: 0.915 allein aus den JPEG-Abmessungen.)")

    # ---- Plot ----
    plot_cols = [f for f in NUMERIC if not np.isnan(aucs.get(f, np.nan))]
    fig, axes = plt.subplots(1, len(plot_cols) + 1,
                             figsize=(4 * (len(plot_cols) + 1), 3.6))
    for ax, f in zip(axes, plot_cols):
        s = pd.to_numeric(df[f], errors="coerce")
        for lab_v, name in [(0, "kein Infiltrat"), (1, "Lung Opacity")]:
            ax.hist(s[y == lab_v].dropna(), bins=40, alpha=0.55,
                    label=name, density=True)
        ax.set_title(f"{f}  (AUC {aucs[f]:.3f})")
        ax.legend(fontsize=7)
    for lab_v, name in [(0, "kein Infiltrat"), (1, "Lung Opacity")]:
        axes[-1].hist(prob[y == lab_v], bins=40, alpha=0.55, label=name, density=True)
    axes[-1].set_title(f"Header-Score (AUC {auc:.3f})")
    axes[-1].legend(fontsize=7)
    fig.suptitle(f"RSNA Metadaten-Leak, Modus '{args.mode}': "
                 f"Header allein trennt mit AUC {auc:.3f}")
    fig.tight_layout()
    fig.savefig(args.out / "rsna_metadata_leak.png", dpi=130)

    df.to_csv(args.out / "rsna_metadata_features.csv", index=False)
    print(f"\ngespeichert: {args.out}/rsna_metadata_features.csv, "
          f"rsna_metadata_leak.png, dicom_headers.csv")


if __name__ == "__main__":
    main()
