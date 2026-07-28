"""Schritt 5: Mit dem trainierten U-Net Lungenmasken fuer den Pneumonie-Datensatz erzeugen.

Wichtig:
  * EIGENE Vorverarbeitung des Segmenters. Das U-Net hat Graustufen in [0,1] bei
    256x256 gelernt - OHNE CLAHE und OHNE Per-Bild-Standardisierung aus Phase 1.
    Genau so muessen wir hier fuettern (nicht die Klassifikator-Transform benutzen).
  * DOMAIN-SHIFT. Trainiert wurde auf Montgomery/Shenzhen (Tuberkulose), angewandt
    wird auf den Pneumonie-Datensatz. Deshalb am Ende eine QC-Vorschau ansehen,
    ob die Masken auch hier anatomisch sitzen.

Die Masken werden als PNG (0/255) gecacht, in einer Ordnerstruktur, die den
Bildordner spiegelt: data/chest_xray_masks/{split}/{klasse}/{stamm}.png
Aufloesung 224x224 = passend zur Klassifikator-Eingabe (Schritt 6).

Aufruf:  python -m segmentation.make_masks
"""

import os
import glob

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from segmentation.unet import UNet
from segmentation.mask_refine import refine_mask

CHECKPOINT_PATH = "checkpoints/unet_best.pth"
SRC_ROOT = "data/chest_xray"
MASK_ROOT = "data/chest_xray_masks"
SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]
SEG_SIZE = 256          # so hat das U-Net gelernt
OUT_SIZE = 224          # so braucht es der Klassifikator spaeter
BATCH = 16
EXTS = ("*.jpeg", "*.jpg", "*.png")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    model = UNet(base_ch=32).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    return model


def preprocess(path):
    """Genau wie im Segmenter-Training: Graustufe, 256x256, [0,1]."""
    img = Image.open(path).convert("L")
    img = TF.resize(img, [SEG_SIZE, SEG_SIZE], interpolation=InterpolationMode.BILINEAR)
    return TF.to_tensor(img)                      # [1,256,256], [0,1]


def collect_jobs():
    """Liste (bild_pfad, ausgabe_maskenpfad) fuer alle Bilder."""
    jobs = []
    for split in SPLITS:
        for cls in CLASSES:
            src_dir = os.path.join(SRC_ROOT, split, cls)
            if not os.path.isdir(src_dir):
                continue
            out_dir = os.path.join(MASK_ROOT, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            files = []
            for pat in EXTS:
                files.extend(glob.glob(os.path.join(src_dir, pat)))
            for f in sorted(files):
                stem = os.path.splitext(os.path.basename(f))[0]
                jobs.append((f, os.path.join(out_dir, stem + ".png")))
    return jobs


@torch.no_grad()
def generate():
    model = load_model()
    jobs = collect_jobs()
    print(f"Geraet: {DEVICE} | Bilder gesamt: {len(jobs)}")

    for i in tqdm(range(0, len(jobs), BATCH), desc="Masken"):
        batch = jobs[i:i + BATCH]
        tensors = torch.stack([preprocess(src) for src, _ in batch]).to(DEVICE)
        logits = model(tensors)                                  # [B,1,256,256]
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()[:, 0]  # [B,256,256] bool
        for (_, out_path), pred in zip(batch, preds):
            clean = refine_mask(pred)                            # säubern + Hülle + Symmetrie
            out = (clean.astype(np.uint8) * 255)
            # erst danach auf Klassifikator-Groesse (NEAREST haelt die Maske binaer)
            out = cv2.resize(out, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_NEAREST)
            Image.fromarray(out, mode="L").save(out_path)

    # --- QC-Vorschau: je 3 NORMAL + 3 PNEUMONIA aus dem Testset ---
    save_qc_preview()
    print(f"\nFertig. Masken unter: {MASK_ROOT}")
    print("Bitte segmentation/mask_qc.png ansehen: sitzen die Masken auch auf")
    print("den PNEUMONIE-Bildern anatomisch korrekt (Domain-Shift-Kontrolle)?")


def save_qc_preview():
    import matplotlib.pyplot as plt
    rows = []
    for cls in CLASSES:
        src_dir = os.path.join(SRC_ROOT, "test", cls)
        files = sorted(sum([glob.glob(os.path.join(src_dir, p)) for p in EXTS], []))[:3]
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            rows.append((cls, f, os.path.join(MASK_ROOT, "test", cls, stem + ".png")))

    fig, axes = plt.subplots(len(rows), 2, figsize=(6, 3 * len(rows)))
    for r, (cls, img_path, mask_path) in enumerate(rows):
        img = np.array(Image.open(img_path).convert("L").resize((OUT_SIZE, OUT_SIZE))) / 255.0
        mask = np.array(Image.open(mask_path)) / 255.0
        axes[r, 0].imshow(img, cmap="gray"); axes[r, 0].set_title(cls, fontsize=9)
        axes[r, 1].imshow(img, cmap="gray")
        axes[r, 1].imshow(mask, cmap="Reds", alpha=0.35)
        axes[r, 1].set_title("Maske ueberlagert", fontsize=9)
        axes[r, 0].axis("off"); axes[r, 1].axis("off")
    plt.tight_layout()
    plt.savefig("segmentation/mask_qc.png", dpi=90)


if __name__ == "__main__":
    generate()
