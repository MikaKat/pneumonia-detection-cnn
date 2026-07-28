"""Schritt 1 der Segmentierung: den heruntergeladenen Datensatz PRÜFEN (nicht trainieren).

Datensatz: "Chest Xray Masks and Labels" (Kaggle, nikhilpandey360) = Montgomery +
Shenzhen mit Lungenmasken, CC0. Beim Entpacken liegen die Bilder in einem Ordner
'CXR_png' und die Masken in 'masks'; manche Maskennamen tragen ein '_mask'-Suffix,
manche nicht, und ~192 Bilder haben GAR KEINE Maske. Damit du nichts von Hand
verschieben musst, sucht dieses Skript Bilder und Masken REKURSIV unter DATA_ROOT.

Der häufigste Segmentierungs-Bug ist ein still verrutschtes Bild/Maske-Paar - das
Modell lernt dann Unsinn, ohne dass ein Fehler auftaucht. Dieses Skript beantwortet
vor dem Training drei Fragen:

  1. Hat JEDES Bild eine passende Maske? (Paarung über den Dateinamen-Stamm)
  2. Sind die Masken wirklich binär (nur Hintergrund + Lunge)?
  3. Liegen Bild und Maske übereinander? (visuelle Überlagerung als PNG)

Vorgehen: den Kaggle-Download einfach nach data/lung_seg/ entpacken.

Aufruf:  python -m segmentation.inspect_data
"""

import os
from collections import Counter

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

DATA_ROOT = "data/lung_seg"          # hierhin den Datensatz entpacken
PREVIEW_PATH = "segmentation/inspect_preview.png"
N_PREVIEW = 4                        # so viele Paare in die Vorschau
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _stem(filename: str) -> str:
    """Dateiname ohne Endung und ohne '_mask'-Suffix -> Paarungsschlüssel."""
    name = os.path.splitext(os.path.basename(filename))[0]
    for suffix in ("_mask", "-mask", "_segmentation"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _is_mask_path(path: str) -> bool:
    """Maske erkannt an einem Ordner 'mask(s)' im Pfad ODER am '_mask'-Suffix."""
    p = path.lower()
    if os.path.basename(os.path.splitext(p)[0]).endswith(("_mask", "-mask", "_segmentation")):
        return True
    parts = p.replace("\\", "/").split("/")
    return any(part in ("mask", "masks") for part in parts)


def _scan(root: str):
    """Läuft rekursiv durch root und teilt alle Bilddateien in Bilder/Masken auf.
    Rückgabe: (images{stamm:pfad}, masks{stamm:pfad})."""
    images, masks = {}, {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(EXTS):
                continue
            full = os.path.join(dirpath, f)
            if _is_mask_path(full):
                masks[_stem(f)] = full
            else:
                images[_stem(f)] = full
    return images, masks


def main():
    if not os.path.isdir(DATA_ROOT):
        print(f"!! {DATA_ROOT} existiert nicht. Kaggle-Datensatz 'Chest Xray Masks")
        print("   and Labels' herunterladen und dorthin entpacken.")
        return

    images, masks = _scan(DATA_ROOT)
    print(f"Bilder gefunden:  {len(images):4d}")
    print(f"Masken gefunden:  {len(masks):4d}   (rekursiv unter {DATA_ROOT})")

    if not images or not masks:
        print("\n!! Keine Bilder oder keine Masken erkannt. Ist der Datensatz")
        print("   vollständig entpackt? Erwartet werden Ordner wie 'CXR_png' und 'masks'.")
        return

    # --- 1. Paarung ---
    paired = sorted(set(images) & set(masks))
    only_img = sorted(set(images) - set(masks))
    only_mask = sorted(set(masks) - set(images))
    print(f"\nVollständige Paare: {len(paired)}")
    print(f"  Bild OHNE Maske: {len(only_img)}  (erwartet ~192 - die trainieren wir nicht mit)")
    if only_img[:3]:
        print(f"    z.B.: {only_img[:3]}")
    if only_mask:
        print(f"  Maske OHNE Bild: {len(only_mask)}  z.B.: {only_mask[:3]}")

    if not paired:
        print("\n!! Keine Paare - die Dateinamen-Stämme passen nicht zusammen.")
        print("   Beispiel Bildnamen:", list(images)[:3])
        print("   Beispiel Maskennamen:", list(masks)[:3])
        return

    # --- 2. Masken-Werte und Größen-Check (Stichprobe) ---
    value_counter = Counter()
    size_mismatch = 0
    checked = 0
    for stem in paired[:200]:                     # Stichprobe reicht
        img = Image.open(images[stem])
        msk = Image.open(masks[stem]).convert("L")
        if img.size != msk.size:
            size_mismatch += 1
        uniq = np.unique(np.array(msk))
        value_counter[len(uniq)] += 1             # binär -> 2 Werte erwartet
        checked += 1

    print(f"\nMasken-Stichprobe geprüft: {checked}")
    print(f"  Bild/Maske gleiche Größe: {checked - size_mismatch}/{checked}"
          f"  (Abweichungen: {size_mismatch})")
    print(f"  Anzahl verschiedener Grauwerte je Maske: "
          f"{dict(sorted(value_counter.items()))}")
    print("  -> erwartet: die meisten Masken mit 2 Werten (0 = Hintergrund, 255 = Lunge).")
    print("     Viele Werte = evtl. keine echte Maske, sondern ein Graubild.")

    # --- 3. Visuelle Überlagerung ---
    n = min(N_PREVIEW, len(paired))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for r, stem in enumerate(paired[:n]):
        img = Image.open(images[stem]).convert("L").resize((256, 256))
        msk = Image.open(masks[stem]).convert("L").resize((256, 256))
        img_a = np.array(img) / 255.0
        msk_a = (np.array(msk) > 127).astype(float)          # binarisieren

        axes[r, 0].imshow(img_a, cmap="gray");  axes[r, 0].set_title(f"Bild: {stem}", fontsize=9)
        axes[r, 1].imshow(msk_a, cmap="gray");  axes[r, 1].set_title("Maske", fontsize=9)
        axes[r, 2].imshow(img_a, cmap="gray")
        axes[r, 2].imshow(msk_a, cmap="Reds", alpha=0.35)    # Maske rot über das Bild
        axes[r, 2].set_title("Überlagerung", fontsize=9)
        for c in range(3):
            axes[r, c].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(PREVIEW_PATH), exist_ok=True)
    plt.savefig(PREVIEW_PATH, dpi=90)
    print(f"\nVorschau gespeichert: {PREVIEW_PATH}")
    print("Bitte ansehen: liegt die rote Maske sauber auf den Lungen? "
          "Wenn ja, ist der Datensatz bereit für Schritt 2.")


if __name__ == "__main__":
    main()
