"""
Trennt beim Schaerfe-Unterschied zwischen NORMAL und PNEUMONIA das
Resampling-Artefakt vom echten radiologischen Befund.

Das Problem: NORMAL-Aufnahmen sind im Kermany-Datensatz systematisch groesser
(siehe metadata_leak_check.py). Nach dem Crop auf eine feste Kantenlaenge
wurden sie also staerker heruntergerechnet als PNEUMONIA-Aufnahmen. Ein
gemessener Schaerfe-Unterschied kann deshalb zweierlei bedeuten:

  (a) Artefakt  -- unterschiedlich starkes Downsampling
  (b) Befund    -- eine Konsolidierung loescht die feine Gefaesszeichnung,
                   das Parenchym wird homogener

Unterscheiden laesst sich das nur, indem man auf die Crop-Groesse MATCHT:
innerhalb eines schmalen crop_side-Bandes wurden beide Klassen gleich stark
skaliert, ein dort verbleibender Unterschied kann kein Skalierungsartefakt
mehr sein.

Gemessen wird die Standardabweichung eines Hochpasses (Bild minus Gauss),
und zwar NUR innerhalb der Lungenmaske -- sonst dominiert der schwarze
Padding-Rand.

Voraussetzung: lung_preprocess.py wurde mit --save-masks gelaufen.

CLI:
  python sharpness_leak_check.py --prepared data/prepared --out qc/
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
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

CLASSES = ("NORMAL", "PNEUMONIA")
MIN_PER_CLASS = 25          # ab wann ein Band ausgewertet wird


def rank_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Richtungsunabhaengiger AUC (Mann-Whitney-U auf Raengen)."""
    y = np.asarray(y)
    s = np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    a = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return max(a, 1 - a)


def sharpness(img: np.ndarray, mask: np.ndarray, sigma: float) -> float:
    """Streuung des Hochpasses innerhalb der Maske."""
    return float((img - gaussian_filter(img, sigma))[mask].std())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    p.add_argument("--variant", default="crop")
    p.add_argument("--out", type=Path, default=Path("qc"))
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--bands", type=int, default=6,
                   help="Anzahl crop_side-Baender (Quantile) fuers Matching")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    log = pd.read_csv(args.prepared / "crop_log.csv")
    log["key"] = log["file"].str.rsplit(".", n=1).str[0]
    log = log.set_index("key")

    root = args.prepared / args.variant
    files = sorted(f for f in root.rglob("*.png"))
    if not files:
        raise SystemExit(f"Keine Bilder unter {root}")

    rows = []
    for f in tqdm(files, desc="schaerfe", unit="img", smoothing=0.05):
        rel = f.relative_to(root)
        key = str(rel.with_suffix("")).replace("\\", "/")
        if key not in log.index:
            continue
        mpath = args.prepared / "mask_repaired" / rel
        if not mpath.exists():
            continue
        img = np.asarray(Image.open(f), dtype=np.float32) / 255.0
        m = np.asarray(Image.open(mpath)) > 127
        if m.sum() < 1000:
            continue
        rows.append({
            "file": key,
            "label": 1 if rel.parent.name == CLASSES[1] else 0,
            "sharp": sharpness(img, m, args.sigma),
            "crop_side": float(log.loc[key, "crop_side"]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "sharpness.csv", index=False)
    y = df["label"].values
    print(f"\nn={len(df)} | NORMAL {int((y == 0).sum())} | PNEUMONIA {int(y.sum())}")

    auc_raw = rank_auc(y, df["sharp"].values)
    auc_cs = rank_auc(y, df["crop_side"].values)
    print(f"\nungematcht   Schaerfe-AUC {auc_raw:.3f}")
    print(f"Kontrolle    crop_side-AUC {auc_cs:.3f}   (das ist der Confounder)")

    # ---- Matching: innerhalb eines Bandes ist die Skalierung vergleichbar ----
    edges = np.quantile(df["crop_side"], np.linspace(0, 1, args.bands + 1))
    edges[-1] += 1
    print(f"\ngematcht auf crop_side ({args.bands} Quantilbaender, "
          f"ausgewertet ab {MIN_PER_CLASS} pro Klasse):")
    print(f"{'Band':<16}{'NORMAL':>8}{'PNEU':>8}{'AUC':>12}")
    print("-" * 44)
    used = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = df[(df["crop_side"] >= lo) & (df["crop_side"] < hi)]
        n0 = int((s["label"] == 0).sum())
        n1 = int(s["label"].sum())
        band = f"{lo:.0f}-{hi:.0f}"
        if min(n0, n1) < MIN_PER_CLASS:
            print(f"{band:<16}{n0:>8}{n1:>8}{'zu duenn':>12}")
            continue
        a = rank_auc(s["label"].values, s["sharp"].values)
        used.append((len(s), a))
        print(f"{band:<16}{n0:>8}{n1:>8}{a:>12.3f}")

    if used:
        w = sum(n for n, _ in used)
        auc_matched = sum(n * a for n, a in used) / w
        print(f"\ngewichtetes Mittel gematcht: {auc_matched:.3f}")
        print(f"davon Artefakt (Differenz zu ungematcht): {auc_raw - auc_matched:+.3f}")
        print("\nLesart: was nach dem Matching uebrig bleibt, kann kein")
        print("Skalierungsartefakt sein -- dort wurden beide Klassen gleich")
        print("stark verkleinert. Das ist der Anteil, der auf Konsolidierung")
        print("zurueckgehen kann und den man NICHT wegnormalisieren sollte.")
    else:
        auc_matched = float("nan")
        print("\nKein Band mit genug Bildern beider Klassen -- die Verteilungen")
        print("ueberlappen zu wenig. Das ist selbst schon ein Befund.")

    # ---- Plot ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for lab, name in [(0, CLASSES[0]), (1, CLASSES[1])]:
        ax[0].hist(df.loc[y == lab, "sharp"], bins=60, alpha=0.55,
                   label=name, density=True)
        ax[1].scatter(df.loc[y == lab, "crop_side"], df.loc[y == lab, "sharp"],
                      s=4, alpha=0.35, label=name)
    ax[0].set_xlabel(f"Hochpass-Streuung in der Lunge (sigma={args.sigma})")
    ax[0].set_title(f"ungematcht: AUC {auc_raw:.3f}")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("crop_side (Originalpixel)")
    ax[1].set_ylabel("Schaerfe")
    ax[1].set_title(f"gematcht: AUC {auc_matched:.3f}")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "sharpness_leak.png", dpi=130)
    print(f"\ngespeichert: {args.out}/sharpness.csv, sharpness_leak.png")


if __name__ == "__main__":
    main()
