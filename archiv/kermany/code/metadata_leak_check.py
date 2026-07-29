"""
Prueft, ob schon die BILDABMESSUNGEN das Label verraten -- ohne einen
einzigen Pixel anzusehen.

Hintergrund: NORMAL- und PNEUMONIA-Aufnahmen im Kermany-Datensatz stammen
aus unterschiedlichen Aufnahmesituationen (Geraet, Zuschnitt, Alter der
Kinder). Wenn sich das schon in Breite/Hoehe der JPEGs niederschlaegt, dann
hat JEDES Modell einen Abkuerzungsweg, der nichts mit Lungenpathologie zu
tun hat:

  * Ein systematischer Groessenunterschied wird beim Resize auf 224x224 zu
    einem systematischen Unterschied im Downsampling-Faktor -- also in der
    Texturschaerfe. Das ist genau der globale Textur-Bias aus Phase 1.
  * Ein systematischer Unterschied im Seitenverhaeltnis wird beim Stretchen
    auf ein Quadrat zu einer systematisch anderen Lungenform -- also genau
    dem "bbox_aspect"-Leak, den mask_leakage_check.py in den Masken findet.

Der hier gemessene AUC ist die Untergrenze fuer den Confounder. Ist er hoch,
sind alle Modellzahlen auf diesem Datensatz nur unter Vorbehalt zu lesen.

Es werden nur die JPEG-Header gelesen (PIL dekodiert nicht), das laeuft
deshalb in Sekunden.

CLI:
  python metadata_leak_check.py --images data/chest_xray --out qc/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

EXTS = {".png", ".jpg", ".jpeg"}
SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")
FEATURES = ["width", "height", "aspect", "n_pixels"]


def patient_group(name: str, split: str, cls: str) -> str:
    """Patienten-ID aus dem Dateinamen -- IDs sind pro Split/Klasse neu vergeben."""
    m = re.search(r"person[_\-]?(\d+)", name, re.I)
    if m:
        return f"{split}|{cls}|person{m.group(1)}"
    m = re.search(r"IM-?(\d+)", name, re.I)
    return f"{split}|{cls}|im{m.group(1)}" if m else f"{split}|{cls}|{name}"


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for cls in CLASSES:
            d = root / split / cls
            if not d.is_dir():
                continue
            for f in sorted(x for x in d.iterdir() if x.suffix.lower() in EXTS):
                w, h = Image.open(f).size          # nur Header, kein Dekodieren
                rows.append({
                    "file": f.name, "split": split, "label": 1 if cls == CLASSES[1] else 0,
                    "group": patient_group(f.name, split, cls),
                    "width": float(w), "height": float(h),
                    "aspect": w / h, "n_pixels": float(w) * h,
                })
    return pd.DataFrame(rows)


def single_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Richtungsunabhaengiger AUC eines einzelnen Merkmals."""
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, default=Path("data/chest_xray"))
    p.add_argument("--out", type=Path, default=Path("qc"))
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = collect(args.images)
    if df.empty:
        raise SystemExit(f"Keine Bilder unter {args.images}/{{train,val,test}}/{{NORMAL,PNEUMONIA}}")
    df.to_csv(args.out / "metadata_features.csv", index=False)

    y = df["label"].values
    print(f"\n{len(df)} Bilder | NORMAL {int((y == 0).sum())} | PNEUMONIA {int(y.sum())}")
    print(f"{len(df['group'].unique())} Patientengruppen\n")

    # ---- Einzelmerkmale, gesamt ----
    print(f"{'Merkmal':<12}{'NORMAL':>12}{'PNEUMONIA':>12}{'AUC':>8}")
    print("-" * 44)
    for f in FEATURES:
        print(f"{f:<12}{df.loc[y == 0, f].median():>12.3f}"
              f"{df.loc[y == 1, f].median():>12.3f}{single_auc(y, df[f].values):>8.3f}")

    # ---- Einzelmerkmale, pro Split ----
    print(f"\n{'Split':<8}{'n':>7}" + "".join(f"{f:>10}" for f in FEATURES))
    print("-" * (15 + 10 * len(FEATURES)))
    for s in SPLITS:
        sub = df[df["split"] == s]
        if sub["label"].nunique() < 2:
            continue
        ys = sub["label"].values
        print(f"{s:<8}{len(sub):>7}" +
              "".join(f"{single_auc(ys, sub[f].values):>10.3f}" for f in FEATURES))

    # ---- Kombiniert, patientengruppierte CV ----
    g = df["group"].values
    X = df[FEATURES].values
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=0)
    prob = cross_val_predict(GradientBoostingClassifier(random_state=0), X, y,
                             groups=g, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, prob)
    print(f"\n>> Nur-Abmessungen-Klassifikator, {args.folds}-fold gruppierte CV AUC = {auc:.3f}")
    print("   Das ist die Baseline, die JEDES Bildmodell auf diesem Datensatz")
    print("   schlagen muss, um ueberhaupt etwas Radiologisches gelernt zu haben.")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, f in zip(axes, FEATURES):
        for lab, name in [(0, "NORMAL"), (1, "PNEUMONIA")]:
            ax.hist(df.loc[y == lab, f], bins=50, alpha=0.55, label=name, density=True)
        ax.set_title(f"{f}  (AUC {single_auc(y, df[f].values):.3f})")
        ax.legend(fontsize=7)
    fig.suptitle(f"Metadaten-Leak: nur Abmessungen, kombinierter AUC {auc:.3f}")
    fig.tight_layout()
    fig.savefig(args.out / "metadata_leak.png", dpi=130)
    print(f"\ngespeichert: {args.out}/metadata_features.csv, metadata_leak.png")


if __name__ == "__main__":
    main()
