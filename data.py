"""Datenladen für die Chest X-Ray Pneumonie-Klassifikation."""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split


class CLAHE:
    """Contrast Limited Adaptive Histogram Equalization.

    Egalisiert den Kontrast LOKAL (in Kacheln) statt global. Das nimmt dem Modell
    den globalen Kontrast-/Helligkeits-Shortcut, den die Diagnose gefunden hat.
    Arbeitet auf einem PIL-Bild und gibt ein PIL-Bild zurück (3 identische Kanäle,
    damit ResNet18 unverändert bleibt).

    clip_limit: begrenzt die Kontrastverstärkung pro Kachel (verhindert Rausch-Boost).
    tile_grid_size: Anzahl Kacheln (8x8 = Standard für 224x224).
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        # cv2-Objekt hier erzeugen (nicht im __init__), damit die Klasse auch mit
        # DataLoader-Workern (num_workers>0) sauber pickelbar bleibt.
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        gray = np.array(img.convert("L"))          # -> 1 Kanal, uint8
        equalized = clahe.apply(gray)              # CLAHE anwenden
        rgb = np.stack([equalized] * 3, axis=-1)   # 3 identische Kanäle
        return Image.fromarray(rgb)


class PerImageStandardize:
    """Standardisiert jedes Bild auf Mittelwert 0 / Std 1 (über alle Pixel).

    Entfernt globale Helligkeits- UND Kontrastunterschiede ZWISCHEN Bildern -
    also genau den globalen Kontrast-Confounder aus der Diagnose. Danach ist die
    globale Kontrast-Zahl (std) für jedes Bild ~1 und kann die Klassen nicht mehr
    trennen. Ersetzt die feste ImageNet-Normalisierung (mean/std).
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, tensor):
        return (tensor - tensor.mean()) / (tensor.std() + self.eps)


DATA_DIR = "data/chest_xray"
BATCH_SIZE = 32
VAL_SIZE = 100          # Anzahl Bilder, die vom Trainingsset für die Validierung abgezweigt werden
RANDOM_STATE = 42        # macht den Split reproduzierbar

# Transformationen: Bildgröße anpassen, lokale Struktur betonen (CLAHE),
# in Tensor umwandeln, per Bild standardisieren (entfernt globalen Kontrast-Confounder)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),   # lokale Struktur betonen
    transforms.ToTensor(),
    PerImageStandardize(),                          # globalen Confounder entfernen
])


def get_data_loaders():
    """Lädt Trainings-, Validierungs- und Test-Daten und gibt die passenden
    DataLoader sowie die Klassennamen zurück."""
    train_data = datasets.ImageFolder(DATA_DIR + "/train", transform=transform)
    test_data = datasets.ImageFolder(DATA_DIR + "/test", transform=transform)

    # Der Val-Ordner des Original-Datensatzes hat nur 16 Bilder -> zu klein für eine
    # verlässliche Validierung. Stattdessen zweigen wir stratifiziert (gleiche
    # Klassen-Verteilung) einen Teil des Trainingssets ab.
    targets = train_data.targets
    train_idx, val_idx = train_test_split(
        list(range(len(train_data))),
        test_size=VAL_SIZE,
        stratify=targets,
        random_state=RANDOM_STATE
    )
    train_subset = Subset(train_data, train_idx)
    val_subset = Subset(train_data, val_idx)

    # --- Balancierter Sampler gegen die 2.9:1-Imbalance ---
    # Labels der Trainings-Teilmenge, in derselben Reihenfolge wie train_subset:
    train_targets = [targets[i] for i in train_idx]
    class_counts = np.bincount(train_targets)          # [NORMAL, PNEUMONIA]
    class_weights = 1.0 / class_counts                 # seltene Klasse -> höheres Gewicht
    sample_weights = [class_weights[t] for t in train_targets]  # Gewicht pro Bild
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),   # 1 Epoche = so viele Ziehungen wie Trainingsbilder
        replacement=True                   # NÖTIG, damit NORMAL überabgetastet werden kann
    )

    # Beim Sampler KEIN shuffle=True (schließt sich gegenseitig aus; der Sampler
    # randomisiert bereits). Val/Test bleiben unverändert.
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, train_data.classes


if __name__ == "__main__":
    # Datei direkt ausführen, um die Datengrößen zu prüfen (kein Training)
    train_loader, val_loader, test_loader, classes = get_data_loaders()
    print("Klassen:", classes)
    print("Anzahl Trainingsbilder:", len(train_loader.dataset))
    print("Anzahl Validierungsbilder:", len(val_loader.dataset))
    print("Anzahl Testbilder:", len(test_loader.dataset))
