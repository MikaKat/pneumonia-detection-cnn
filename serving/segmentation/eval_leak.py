"""Schnelle Messung des Formen-Leaks OHNE Neutraining.

lung_area ist modellunabhängig - es hängt nur an den Masken. Deshalb prüfen wir
hier die Wirkung der Masken-Verfeinerung (mask_refine.refine_mask) auf die
lung_area-AUC, BEVOR wir v4 überhaupt neu trainieren. So iterieren wir die
Verfeinerung (Flags in mask_refine.py) in Sekunden statt in Trainingsläufen.

Vergleicht die gespeicherten Test-Masken VORHER vs. NACHHER refine_mask:
  * mittlerer Lungenflächen-Anteil je Klasse
  * AUC des Merkmals lung_area (Ziel: näher an 0.5 = weniger Leak)

Aufruf:  python -m segmentation.eval_leak
"""

import os
import glob

import numpy as np
from PIL import Image

from segmentation.mask_refine import refine_mask

MASK_DIR = "data/chest_xray_masks/test"
CLASSES = ["NORMAL", "PNEUMONIA"]


def auc(pos, neg):
    """AUC von 'pos' gegen 'neg' (Mann-Whitney-U, wie in diagnostics.py)."""
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


def main():
    raw = {c: [] for c in CLASSES}
    ref = {c: [] for c in CLASSES}
    for cls in CLASSES:
        files = sorted(glob.glob(os.path.join(MASK_DIR, cls, "*.png")))
        for f in files:
            m = (np.array(Image.open(f).convert("L")) > 127).astype(np.uint8)
            raw[cls].append(float(m.mean()))
            ref[cls].append(float(refine_mask(m).mean()))
    print(f"Test-Masken: NORMAL {len(raw['NORMAL'])} | PNEUMONIA {len(raw['PNEUMONIA'])}\n")

    for name, d in [("VORHER (gespeichert)", raw), ("NACHHER (refine_mask)", ref)]:
        mn = np.mean(d["NORMAL"]); mp = np.mean(d["PNEUMONIA"])
        a = auc(np.array(d["PNEUMONIA"]), np.array(d["NORMAL"]))
        print(f"{name:<24} mean NORMAL {mn:.3f} | mean PNEU {mp:.3f} | "
              f"lung_area AUC {a:.3f} | Trennkraft {abs(a - 0.5) * 2:.3f}")

    print("\nZiel: AUC näher an 0.5 (weniger Formen-Leak). Wenn refine_mask hilft,")
    print("Masken mit  python -m segmentation.make_masks  neu erzeugen, dann v4 neu trainieren.")


if __name__ == "__main__":
    main()
