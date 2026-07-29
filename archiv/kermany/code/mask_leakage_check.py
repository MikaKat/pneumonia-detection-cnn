"""
Prueft, ob die Maskenform selbst das Label verraet.

Idee: wenn das Segmentierungsnetz bei PNEUMONIA systematisch mehr wegschneidet,
dann kann ein Klassifikator "Maskenform -> Label" lernen, ohne je das Roentgenbild
anzusehen. Dieses Skript trainiert genau so einen Klassifikator -- aber NUR auf
Form-Features, ohne Bildpixel. Der resultierende AUC ist deine Obergrenze fuer
Leakage.

  AUC ~ 0.50-0.60  -> unkritisch
  AUC ~ 0.60-0.70  -> Vorsicht, Masken vorher reparieren
  AUC > 0.70       -> hartes Maskieren in dieser Form nicht verwenden

Erwartete Ordnerstruktur (Klassenname = direkter Elternordner):
  masks/NORMAL/*.png
  masks/PNEUMONIA/*.png
oder, wie in diesem Projekt:
  data/chest_xray_masks/{train,val,test}/{NORMAL,PNEUMONIA}/*.png

CLI:
  python mask_leakage_check.py --masks data/chest_xray_masks --out qc/
  python mask_leakage_check.py --masks data/chest_xray_masks/test --out qc/test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import mannwhitneyu
from skimage.measure import label, regionprops
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

FEATURES = [
    "area_frac", "solidity", "n_components", "lr_ratio",
    "bbox_aspect", "centroid_y", "extent",
]


def mask_features(mask: np.ndarray) -> dict:
    """Reine Form-Features, keine Bildinformation."""
    m = mask.astype(bool)
    h, w = m.shape
    out = {f: np.nan for f in FEATURES}
    out["area_frac"] = m.sum() / m.size
    if not m.any():
        return out

    lbl = label(m)
    props = sorted(regionprops(lbl), key=lambda r: r.area, reverse=True)
    out["n_components"] = float(len(props))

    # Solidity ueber die Gesamtmaske: faellt ab, sobald eine Verschattung
    # aus der Lunge herausgeschnitten wurde
    ys, xs = np.where(m)
    total = regionprops(m.astype(np.uint8))[0]
    out["solidity"] = float(total.solidity)
    out["extent"] = float(total.extent)
    out["centroid_y"] = float(ys.mean() / h)
    out["bbox_aspect"] = float((xs.max() - xs.min() + 1) / max(1, ys.max() - ys.min() + 1))

    # Flaechenverhaeltnis der beiden groessten Komponenten (Seitenasymmetrie)
    if len(props) >= 2:
        out["lr_ratio"] = float(props[1].area / props[0].area)
    else:
        out["lr_ratio"] = 0.0
    return out


def collect(mask_dir: Path) -> pd.DataFrame:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    rows = []
    for f in sorted(x for x in mask_dir.rglob("*") if x.suffix.lower() in exts):
        rel = f.relative_to(mask_dir)
        # Klasse = direkter Elternordner. Funktioniert sowohl fuer
        #   masks/NORMAL/*.png
        # als auch fuer das Projektlayout
        #   data/chest_xray_masks/{split}/{NORMAL,PNEUMONIA}/*.png
        cls = f.parent.name if len(rel.parts) > 1 else "UNKNOWN"
        m = np.array(Image.open(f).convert("L")) > 127
        row = {"file": str(rel), "label": cls}
        row.update(mask_features(m))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--masks", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("qc"))
    p.add_argument("--positive", default="PNEUMONIA")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = collect(args.masks)
    if df.empty:
        raise SystemExit(f"Keine Masken unter {args.masks}")
    df.to_csv(args.out / "mask_features.csv", index=False)

    classes = sorted(df["label"].unique())
    print(f"\n{len(df)} Masken, Klassen: {dict(df['label'].value_counts())}\n")

    if len(classes) != 2:
        print("Genau zwei Klassen noetig fuer den Leakage-Test.")
        return

    y = (df["label"] == args.positive).astype(int).values
    X = df[FEATURES].fillna(df[FEATURES].median()).values

    # ---- Einzelfeature-Vergleich ----
    print(f"{'Feature':<15}{'NORMAL':>12}{'PNEUMONIA':>12}{'p':>12}{'AUC':>8}")
    print("-" * 59)
    for f in FEATURES:
        a = df.loc[y == 0, f].dropna()
        b = df.loc[y == 1, f].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        _, pv = mannwhitneyu(a, b)
        auc = roc_auc_score(y, df[f].fillna(df[f].median()))
        print(f"{f:<15}{a.median():>12.3f}{b.median():>12.3f}{pv:>12.2e}{max(auc, 1-auc):>8.3f}")

    # ---- Kombinierter Klassifikator, nur Form ----
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    prob = cross_val_predict(
        GradientBoostingClassifier(random_state=0), X, y, cv=cv, method="predict_proba"
    )[:, 1]
    auc = roc_auc_score(y, prob)
    print(f"\n>> Nur-Form-Klassifikator, 5-fold CV AUC = {auc:.3f}")
    if auc > 0.70:
        print("   ACHTUNG: starkes Leakage. Hartes Maskieren hier nicht verwenden.")
    elif auc > 0.60:
        print("   Grenzwertig. Masken reparieren und erneut pruefen.")
    else:
        print("   Unkritisch.")

    # ---- Plots ----
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, f in zip(axes.ravel(), FEATURES):
        for c in classes:
            ax.hist(df.loc[df["label"] == c, f].dropna(), bins=30,
                    alpha=0.55, label=c, density=True)
        ax.set_title(f)
        ax.legend(fontsize=7)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(0.05, 0.5, f"Nur-Form-AUC\n{auc:.3f}", fontsize=15)
    fig.tight_layout()
    fig.savefig(args.out / "mask_feature_distributions.png", dpi=130)
    print(f"\ngespeichert: {args.out}/mask_features.csv, mask_feature_distributions.png")


if __name__ == "__main__":
    main()
