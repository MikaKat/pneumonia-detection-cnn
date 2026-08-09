"""Schritt 4: U-Net trainieren (Dice+BCE-Loss, Dice/IoU-Metrik).

Zwei Kernideen:
  * LOSS = BCE + Dice. Die Lunge ist ~25 % der Pixel (Ungleichgewicht wie in
    Phase 1, nur pro Pixel). BCE gibt stabile Pixel-Gradienten, ist aber vom
    Hintergrund dominierbar; Dice optimiert direkt die Flaechen-Ueberlappung und
    ist gegen das Ungleichgewicht robust. Zusammen ergaenzen sie sich.
  * METRIK != Loss. Zum Bewerten binarisieren wir die Vorhersage (Sigmoid > 0.5)
    und messen Dice und IoU (Ueberlappung mit der wahren Maske).

Anders als in Phase 1 wird das GANZE Netz trainiert (kein eingefrorenes Backbone).

Achtung: rechenintensiv auf der CPU. Bei zu langer Laufzeit in seg_data.py
IMG_SIZE=128 setzen (ca. 4x schneller) oder EPOCHS reduzieren.

Aufruf:  python -m segmentation.train_unet
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from segmentation.seg_data import get_seg_loaders
from segmentation.unet import UNet

EPOCHS = 20
LEARNING_RATE = 1e-3
CHECKPOINT_PATH = "checkpoints/unet_best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Loss-Bausteine
# ---------------------------------------------------------------------------
def dice_loss(logits, target, eps: float = 1.0):
    """Weicher Dice-Loss auf den Wahrscheinlichkeiten (differenzierbar).

    Nutzt sigmoid(logits) statt harter 0/1 - so bleibt der Gradient fliessend.
    1 - Dice, damit 'kleiner = besser'. eps glaettet und verhindert Division/0.
    """
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)                 # [B, H*W]
    target = target.flatten(1)
    inter = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


bce = nn.BCEWithLogitsLoss()                 # numerisch stabil: Sigmoid intern


def combined_loss(logits, target):
    return bce(logits, target) + dice_loss(logits, target)


# ---------------------------------------------------------------------------
# Metriken (auf BINARISIERTER Vorhersage, nicht differenzierbar - nur zum Messen)
# ---------------------------------------------------------------------------
@torch.no_grad()
def dice_and_iou(logits, target, eps: float = 1e-6):
    pred = (torch.sigmoid(logits) > 0.5).float().flatten(1)
    target = target.flatten(1)
    inter = (pred * target).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    target_sum = target.sum(dim=1)
    union = pred_sum + target_sum - inter
    dice = (2 * inter + eps) / (pred_sum + target_sum + eps)
    iou = (inter + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()


def run_epoch(model, loader, optimizer=None):
    """Eine Epoche. optimizer=None -> Validierung (kein Backprop)."""
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss = total_dice = total_iou = 0.0
    phase = "Train" if train else "Val"

    for imgs, masks in tqdm(loader, desc=phase, leave=False):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        with torch.set_grad_enabled(train):
            logits = model(imgs)
            loss = combined_loss(logits, masks)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        d, i = dice_and_iou(logits, masks)
        total_loss += loss.item(); total_dice += d; total_iou += i

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def train():
    train_loader, val_loader = get_seg_loaders()
    print(f"Geraet: {DEVICE} | Train-Paare: {len(train_loader.dataset)} | "
          f"Val-Paare: {len(val_loader.dataset)}")

    model = UNet(base_ch=32).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Bei stagnierendem Val-Dice die Lernrate automatisch senken:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                     factor=0.5, patience=3)

    os.makedirs("checkpoints", exist_ok=True)
    best_dice = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_dice, tr_iou = run_epoch(model, train_loader, optimizer)
        va_loss, va_dice, va_iou = run_epoch(model, val_loader)
        scheduler.step(va_dice)

        print(f"Epoche {epoch:2d}/{EPOCHS} | "
              f"Train Loss {tr_loss:.4f} Dice {tr_dice:.4f} | "
              f"Val Loss {va_loss:.4f} Dice {va_dice:.4f} IoU {va_iou:.4f}")

        # Bestes Modell nach Val-DICE speichern (nicht nach Loss - Dice ist die Zielmetrik)
        if va_dice > best_dice:
            best_dice = va_dice
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> neues bestes Modell gespeichert (Val Dice {va_dice:.4f})")

    print(f"\nFertig. Bester Val-Dice: {best_dice:.4f}  ->  {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
