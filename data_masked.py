"""Schritt 6 (Daten): Klassifikator-Eingabe auf die Lunge maskieren.

Baut auf data.py auf, ersetzt aber die Vorverarbeitung: Zusaetzlich zu CLAHE wird
die (dilatierte) Lungenmaske aus segmentation/make_masks.py angewandt. Alles
ausserhalb der Lunge wird auf 0 gesetzt, damit der globale Helligkeits-/Kontrast-
und Rand-Confounder aus der Phase-1-Diagnose verschwindet.

Zwei entscheidende Details:
  1. DILATATION: die Maske wird um einen Sicherheitsrand erweitert (DILATE_PX),
     damit Untersegmentierung an der Lungengrenze kein echtes Gewebe abschneidet.
  2. STANDARDISIERUNG NUR UEBER LUNGEN-PIXEL: sonst haengen mean/std von der
     Lungenflaeche ab - und die korreliert mit der Klasse (Segmenter unterschaetzt
     Verschattung). Das waere ein NEUER Shortcut. Deshalb Statistik nur aus der
     Lunge, dann Hintergrund auf 0 (= Lungen-Mittelwert nach Standardisierung).

Ansonsten identisch zu data.py (gleicher Val-Split, gleicher Sampler) - damit der
Vergleich v3 (unmaskiert) vs. v4 (maskiert) sauber bleibt.

Selbsttest:  python -m data_masked  (druckt Groessen, speichert masked_preview.png)
"""

import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from data import CLAHE, DATA_DIR, BATCH_SIZE, VAL_SIZE, RANDOM_STATE

MASK_DIR = "data/chest_xray_masks"
DILATE_PX = 15          # Sicherheitsrand der Maske (px auf 224) gegen Untersegmentierung
EPS = 1e-6

_resize = transforms.Resize((224, 224))
_clahe = CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
_dilate_kernel = np.ones((DILATE_PX, DILATE_PX), np.uint8)


def _mask_path(img_path: str) -> str:
    """data/chest_xray/<split>/<klasse>/x.jpeg -> data/chest_xray_masks/.../x.png"""
    rel = os.path.relpath(img_path, DATA_DIR)
    stem = os.path.splitext(rel)[0]
    return os.path.join(MASK_DIR, stem + ".png")


def load_raw_mask(img_path: str) -> np.ndarray:
    """Gespeicherte Lungenmaske (224x224), binarisiert, OHNE Dilatation.
    = anatomisches Ziel für die Grad-CAM-Überlappung und den Formen-Leak-Test."""
    mp = _mask_path(img_path)
    if os.path.exists(mp):
        m = np.array(Image.open(mp).convert("L").resize((224, 224), Image.NEAREST))
        return (m > 127).astype(np.uint8)
    return np.ones((224, 224), np.uint8)


def load_dilated_mask(img_path: str) -> np.ndarray:
    """Wie load_raw_mask, aber mit Sicherheitsrand (DILATE_PX) - das, was der
    Klassifikator tatsächlich als Sichtfeld bekommt."""
    m = load_raw_mask(img_path)
    if DILATE_PX > 0 and m.min() == 0:                  # nur wenn es echte Maske gibt
        m = cv2.dilate(m, _dilate_kernel, iterations=1)
    return m


def load_masked_tensor(img_path: str, blur_radius: int = 0) -> torch.Tensor:
    """Bild laden -> CLAHE -> (optional Blur) -> Lungenmaske (dilatiert) anwenden
    -> lungenbasiert standardisieren. Rueckgabe: [3, 224, 224].

    blur_radius > 0 fügt den Blur an derselben Stelle ein wie der Blur-Test der
    Diagnostik (nach CLAHE), damit v4 fair getestet werden kann."""
    img = _resize(Image.open(img_path).convert("RGB"))
    img = _clahe(img)                                   # 3 Kanaele, lokaler Kontrast
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    t = TF.to_tensor(img)                               # [3,224,224], [0,1]

    mask = torch.from_numpy(load_dilated_mask(img_path).astype(np.float32))
    lung = mask > 0.5

    # --- Standardisierung nur ueber Lungen-Pixel, dann Hintergrund auf 0 ---
    if lung.sum() < 100:                                # Sicherung gegen (fast) leere Maske
        return (t - t.mean()) / (t.std() + EPS)         # dann eben global (selten)
    vals = t[:, lung]                                   # [3, N] nur Lungen-Pixel
    mean, std = vals.mean(), vals.std()
    t = (t - mean) / (std + EPS)
    t = t * mask.unsqueeze(0)                           # Hintergrund -> 0
    return t


class MaskedChestXray(Dataset):
    """Liefert (maskierter_bild_tensor, label). samples = Liste (pfad, label)."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return load_masked_tensor(path), label


def get_masked_data_loaders():
    """Wie data.get_data_loaders, aber mit maskierter Eingabe. Gleicher Val-Split
    und gleicher WeightedRandomSampler wie in Phase 1 (Vergleichbarkeit)."""
    train_folder = datasets.ImageFolder(DATA_DIR + "/train")
    test_folder = datasets.ImageFolder(DATA_DIR + "/test")
    classes = train_folder.classes
    targets = train_folder.targets

    train_idx, val_idx = train_test_split(
        list(range(len(train_folder))),
        test_size=VAL_SIZE, stratify=targets, random_state=RANDOM_STATE,
    )
    train_samples = [train_folder.samples[i] for i in train_idx]
    val_samples = [train_folder.samples[i] for i in val_idx]

    # Balancierter Sampler gegen die 2.9:1-Imbalance (wie data.py)
    train_targets = [targets[i] for i in train_idx]
    class_counts = np.bincount(train_targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights),
                                    num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(MaskedChestXray(train_samples), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(MaskedChestXray(val_samples), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(MaskedChestXray(test_folder.samples), batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader, classes


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_loader, val_loader, test_loader, classes = get_masked_data_loaders()
    print("Klassen:", classes)
    print("Train:", len(train_loader.dataset), "| Val:", len(val_loader.dataset),
          "| Test:", len(test_loader.dataset))

    imgs, labels = next(iter(train_loader))
    print(f"Batch: {tuple(imgs.shape)}  min={imgs.min():.2f} max={imgs.max():.2f}")

    # Vorschau: maskierte Eingaben - Hintergrund muss 0 (neutral) sein, Lunge erhalten
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, img, lab in zip(axes.ravel(), imgs, labels):
        ax.imshow(img[0], cmap="gray")
        ax.set_title(classes[lab], fontsize=9); ax.axis("off")
    plt.tight_layout()
    plt.savefig("segmentation/masked_preview.png", dpi=90)
    print("Vorschau gespeichert: segmentation/masked_preview.png")
    print("Pruefen: ist der Hintergrund neutral/leer und die Lunge (mit Rand) erhalten?")
