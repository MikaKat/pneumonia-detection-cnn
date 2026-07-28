"""Schritt 2: Dataset & DataLoader für die Lungensegmentierung.

Liefert Paare (Bild, Maske) statt (Bild, Label). Drei Besonderheiten gegenüber
der Klassifikation aus Phase 1 - alle drei sind klassische Fehlerquellen:

  1. GEKOPPELTE AUGMENTIERUNG: Dreht/spiegelt man das Bild, muss die Maske mit
     EXAKT denselben Zufallsparametern mitgedreht werden - sonst zeigt die Maske
     ins Leere. Deshalb würfeln wir die Werte einmal und wenden sie auf beide an.
  2. MASKE nur mit NEAREST interpolieren: bilinear würde aus 0/1 Zwischenwerte
     (0.37 ...) machen - die Maske muss binär bleiben. Das Bild darf bilinear.
  3. MASKE wird NICHT normalisiert: Bild -> [0,1], Maske bleibt exakt {0,1}.

Selbsttest (kein Training):  python -m segmentation.seg_data
  -> druckt Formen/Wertebereiche und speichert eine Augmentierungs-Vorschau,
     an der man sieht, dass Bild und Maske gemeinsam kippen.
"""

import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from segmentation.inspect_data import _scan, DATA_ROOT

IMG_SIZE = 256          # 256 = guter Kompromiss; 128 = schneller auf der CPU
VAL_FRAC = 0.1          # 10 % der Paare als Validierung
BATCH_SIZE = 8
RANDOM_STATE = 42
MAX_ROT = 10            # Augmentierung: Drehung in Grad (+/-)
USE_HFLIP = True        # Links/Rechts-Spiegelung (für reine Lungenmaske unkritisch)
# Intensitäts-Augmentierung gegen den Domain-Shift (Konsolidierung sieht hell aus):
GAMMA_RANGE = (0.6, 1.6)
BRIGHT_RANGE = (0.7, 1.5)
CONTRAST_RANGE = (0.7, 1.4)
CONSOLIDATION_P = 0.7        # Wahrscheinlichkeit für eine synthetische Verschattung
CONSOLIDATION_MAX_BLOBS = 4  # so viele helle Flecken max.
BLOB_SIGMA = (0.08, 0.22)    # Fleckgröße relativ zur Bildhöhe
BLOB_AMP = (0.3, 0.8)        # Helligkeit der Flecken (bis fast opak)
WHOLE_SIDE_P = 0.35          # Anteil: statt Flecken eine GANZE Lungenseite verdichten


def get_pairs():
    """Liste von (bild_pfad, masken_pfad) für alle vollständig gepaarten Bilder."""
    images, masks = _scan(DATA_ROOT)
    stems = sorted(set(images) & set(masks))
    return [(images[s], masks[s]) for s in stems]


def _intensity_aug(img):
    """Zufällige Helligkeit/Gamma/Kontrast NUR aufs Bild (Maske bleibt gleich).
    Lehrt das U-Net, dass dieselbe Lunge bei anderer Belichtung Lunge bleibt."""
    if random.random() < 0.8:
        img = TF.adjust_gamma(img, random.uniform(*GAMMA_RANGE))
    if random.random() < 0.8:
        img = TF.adjust_brightness(img, random.uniform(*BRIGHT_RANGE))
    if random.random() < 0.8:
        img = TF.adjust_contrast(img, random.uniform(*CONTRAST_RANGE))
    return img.clamp(0, 1)


def _add_consolidation(img, mask):
    """Simuliert eine Pneumonie-Verschattung: helle Bereiche INNERHALB der Lunge,
    während die Maske unverändert bleibt. So lernt das Netz genau den Fehlerfall
    aus der QC - 'hell im Rippenbogen ist trotzdem Lunge'.

    Zwei Varianten:
      * mehrere helle Gauß-Flecken (fokale Konsolidierung), und
      * mit WHOLE_SIDE_P: eine ganze Lungenseite verdichten (lobäre/komplette
        Verschattung - der 'halbe Lunge fehlt'-Fall)."""
    if random.random() > CONSOLIDATION_P:
        return img
    _, H, W = img.shape
    fg = torch.nonzero(mask[0] > 0.5)                  # Lungen-Pixel [N, 2] (y, x)
    if fg.numel() == 0:
        return img

    # Variante A: fokale helle Flecken
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    for _ in range(random.randint(1, CONSOLIDATION_MAX_BLOBS)):
        cy, cx = fg[random.randint(0, fg.shape[0] - 1)].tolist()
        sigma = random.uniform(*BLOB_SIGMA) * H
        amp = random.uniform(*BLOB_AMP)
        blob = amp * torch.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)))
        img = img + blob.unsqueeze(0)

    # Variante B: eine ganze Lungenseite verdichten (Maske bleibt gleich!)
    if random.random() < WHOLE_SIDE_P:
        x_med = int(fg[:, 1].median().item())          # Mediastinum grob = Median-x
        side = mask[0].clone()
        if random.random() < 0.5:
            side[:, x_med:] = 0                        # nur linke Seite behalten
        else:
            side[:, :x_med] = 0                        # nur rechte Seite behalten
        img = img + random.uniform(0.3, 0.7) * side.unsqueeze(0)

    return img.clamp(0, 1)


class LungSegDataset(Dataset):
    """Ein Paar (Bild, Maske) pro Index.

    Rückgabe:
        image: FloatTensor [1, H, W], Werte in [0,1]
        mask:  FloatTensor [1, H, W], Werte in {0,1}
    """

    def __init__(self, pairs, img_size: int = IMG_SIZE, augment: bool = False):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("L")            # 1 Kanal Graustufen
        mask = Image.open(mask_path).convert("L")

        # 1) Größe angleichen: Bild bilinear (weich), Maske NEAREST (bleibt binär)
        img = TF.resize(img, [self.img_size, self.img_size],
                        interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size],
                         interpolation=InterpolationMode.NEAREST)

        # 2) Gekoppelte Augmentierung (nur beim Training): gleiche Parameter für beide
        if self.augment:
            if USE_HFLIP and random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            angle = random.uniform(-MAX_ROT, MAX_ROT)      # EINMAL würfeln ...
            img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            #   ... und auf Bild UND Maske anwenden.

        # 3) In Tensoren: Bild -> [0,1]; Maske -> hart binarisiert {0,1}
        img_t = TF.to_tensor(img)                          # [1,H,W], float [0,1]
        mask_np = (np.array(mask) > 127).astype(np.float32)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0)    # [1,H,W], {0,1}

        # 4) Intensitäts-Augmentierung NUR aufs Bild (Maske unverändert!) -
        #    gegen den Domain-Shift: Helligkeit/Kontrast + synthetische Verschattung.
        if self.augment:
            img_t = _intensity_aug(img_t)
            img_t = _add_consolidation(img_t, mask_t)

        return img_t, mask_t


def get_seg_loaders(img_size: int = IMG_SIZE, batch_size: int = BATCH_SIZE):
    """Teilt die Paare reproduzierbar in Train/Val und baut die DataLoader.
    Augmentierung NUR auf dem Trainingsteil."""
    pairs = get_pairs()
    rng = random.Random(RANDOM_STATE)
    rng.shuffle(pairs)                                     # reproduzierbar mischen
    n_val = int(len(pairs) * VAL_FRAC)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    train_ds = LungSegDataset(train_pairs, img_size, augment=True)
    val_ds = LungSegDataset(val_pairs, img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pairs = get_pairs()
    print(f"Gepaarte Bilder gesamt: {len(pairs)}")

    train_loader, val_loader = get_seg_loaders()
    print(f"Train-Paare: {len(train_loader.dataset)} | Val-Paare: {len(val_loader.dataset)}")

    # Eine Batch ziehen und Formen/Wertebereiche prüfen
    imgs, masks = next(iter(train_loader))
    print(f"Bild-Batch:  {tuple(imgs.shape)}  min={imgs.min():.3f} max={imgs.max():.3f}")
    print(f"Masken-Batch:{tuple(masks.shape)}  Werte={torch.unique(masks).tolist()}")
    print("  -> Bild in [0,1], Maske exakt {0.0, 1.0} = korrekt.")

    # Augmentierungs-Vorschau: dasselbe Paar mehrfach augmentiert.
    # Bild und Maske müssen IMMER GEMEINSAM kippen.
    ds = LungSegDataset(pairs, augment=True)
    fig, axes = plt.subplots(3, 2, figsize=(6, 9))
    for r in range(3):
        img_t, mask_t = ds[0]                              # immer dasselbe Bild, neue Zufallsaugmentierung
        axes[r, 0].imshow(img_t[0], cmap="gray")
        axes[r, 0].imshow(mask_t[0], cmap="Reds", alpha=0.35)
        axes[r, 0].set_title(f"augmentiert #{r+1}", fontsize=9)
        axes[r, 1].imshow(mask_t[0], cmap="gray")
        axes[r, 1].set_title("zugehörige Maske", fontsize=9)
        axes[r, 0].axis("off"); axes[r, 1].axis("off")
    plt.tight_layout()
    plt.savefig("segmentation/augment_preview.png", dpi=90)
    print("Augmentierungs-Vorschau gespeichert: segmentation/augment_preview.png")
    print("Prüfen: kippt die rote Maske in jeder Zeile GEMEINSAM mit dem Bild?")
