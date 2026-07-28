"""
Grad-CAM zum Anschauen -- mit eingezeichneten Bounding Boxes.

Die Zahl steht schon fest (Treffer 0,539 gegen Zufall 0,117, Faktor 4,6 ueber
fuenf Folds). Dieses Skript liefert das Bild dazu, und zwar so, dass es nicht
schmeichelt:

  * Die **Bounding Box wird eingezeichnet**. Eine Heatmap ohne Referenz sieht
    immer plausibel aus -- das Auge findet nachtraeglich einen Grund, warum die
    warme Stelle richtig liegt. Mit Box ist der Vergleich vorgegeben.
  * Das **Maximum der Heatmap wird markiert** (das ist die Groesse, die in die
    Trefferquote eingeht -- nicht der optische Schwerpunkt).
  * Es werden **Treffer UND Fehlschlaege** gezeigt, in getrennten Zeilen und in
    der gemessenen Mischung. Wer nur die gelungenen Beispiele zeigt, berichtet
    eine Trefferquote von 1,0.
  * Optional eine Zeile **Negative** (`--negatives`): dort gibt es keine Box,
    und die Frage ist, ob die Karte diffus bleibt oder trotzdem irgendwohin
    zeigt.

Laeuft auf der CPU, ein paar Sekunden je Bild.

CLI:
  python rsna_gradcam_grid.py --fold 0 --n 6
  python rsna_gradcam_grid.py --fold 3 --n 6 --negatives    # bester CAM-Fold
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from rsna_train import BOX_SPACE, build_transforms, load_boxes, make_model


def compute_cam(cam, tf, root: Path, pid: str, size: int):
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
    img = Image.open(root / f"{pid}.png").convert("L")
    x = tf(img).unsqueeze(0)
    heat = np.clip(cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(1)])[0],
                   0, None)
    with torch.no_grad():
        p = float(torch.sigmoid(cam.model(x).squeeze()).item())
    return np.asarray(img.resize((size, size))), heat, p


def draw(ax, base, heat, boxes, size, title):
    ax.imshow(base, cmap="gray", vmin=0, vmax=255)
    ax.imshow(heat, cmap="jet", alpha=0.42, vmin=0, vmax=max(heat.max(), 1e-9))
    s = size / BOX_SPACE
    for bx, by, bw, bh in boxes:
        ax.add_patch(patches.Rectangle((bx * s, by * s), bw * s, bh * s,
                                       fill=False, edgecolor="lime", lw=2.0))
    y, x = np.unravel_index(int(np.argmax(heat)), heat.shape)
    ax.plot(x, y, "w+", ms=13, mew=2.4)
    ax.plot(x, y, "k+", ms=13, mew=1.0)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, default=Path("data/rsna/png512"))
    p.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--n", type=int, default=6, help="Bilder je Zeile")
    p.add_argument("--negatives", action="store_true",
                   help="dritte Zeile mit Negativen (dort gibt es keine Box)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    ckpt = args.ckpt or Path(f"checkpoints/rsna_f{args.fold}_s{args.seed}.pth")
    out = args.out or Path(f"qc/rsna/gradcam_f{args.fold}.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    sp = json.loads(args.splits.read_text())
    labels = {k: int(v) for k, v in sp["labels"].items()}
    vpmap = sp["viewpos"]
    val = sp["folds"][args.fold]["val"]
    boxes = load_boxes(args.csv)

    model = make_model(torch.device("cpu"))
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    from pytorch_grad_cam import GradCAM
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    tf = build_transforms(args.size, False)
    s = args.size / BOX_SPACE

    # Die Vorauswahl kommt aus der gespeicherten CAM-Auswertung, sofern
    # vorhanden -- so zeigt das Bild dieselben Faelle, die in die Zahl
    # eingegangen sind, statt einer neuen, guenstiger gewuerfelten Stichprobe.
    import pandas as pd
    cam_csv = Path(f"predictions_rsna/cam_f{args.fold}_s{args.seed}.csv")
    rng = np.random.default_rng(args.seed)
    if cam_csv.exists():
        c = pd.read_csv(cam_csv)
        hits = c[c.hit].patientId.tolist()
        miss = c[~c.hit].patientId.tolist()
        print(f"aus {cam_csv.name}: {len(hits)} Treffer, {len(miss)} Fehlschlaege "
              f"({len(hits) / len(c):.3f})")
    else:
        pos = [i for i in val if i in boxes]
        hits, miss = pos, pos
        print("keine CAM-CSV gefunden -- Faelle werden zufaellig gezogen")

    rows = [("Treffer", list(rng.choice(hits, min(args.n, len(hits)), replace=False))),
            ("Fehlschlag", list(rng.choice(miss, min(args.n, len(miss)), replace=False)))]
    if args.negatives:
        neg = [i for i in val if labels[i] == 0]
        rows.append(("negativ (keine Box)",
                     list(rng.choice(neg, min(args.n, len(neg)), replace=False))))

    fig, axes = plt.subplots(len(rows), args.n,
                             figsize=(2.5 * args.n, 2.9 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (name, ids) in enumerate(rows):
        for cix, pid in enumerate(ids):
            base, heat, prob = compute_cam(cam, tf, args.images, pid, args.size)
            bxs = boxes.get(pid, [])
            mask = np.zeros_like(heat, bool)
            for bx, by, bw, bh in bxs:
                mask[max(int(by * s), 0):int((by + bh) * s),
                     max(int(bx * s), 0):int((bx + bw) * s)] = True
            yx = np.unravel_index(int(np.argmax(heat)), heat.shape)
            info = f"p={prob:.2f} {vpmap[pid]}"
            if bxs:
                info += f" | {'TREFFER' if mask[yx] else 'daneben'} | Box {mask.mean():.2f}"
            draw(axes[r, cix], base, heat, bxs, args.size, info)
            if cix == 0:
                axes[r, cix].set_ylabel(name, fontsize=10)
        print(f"  Zeile '{name}' fertig")

    fig.suptitle(f"Grad-CAM, Fold {args.fold} | gruen = Bounding Box, "
                 f"+ = Maximum der Heatmap (das zaehlt fuer die Trefferquote)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\ngespeichert: {out}")


if __name__ == "__main__":
    main()
