"""Gebuendelte Modell-Diagnostik - ein Lauf, ein Ergebnisordner.

Fasst alle Einzeltests zusammen:
  1. CAM-Vergleich  (Grad-CAM vs Grad-CAM++ vs HiResCAM)
  2. Layer-Vergleich (layer4 / layer3 / layer2)
  3. Occlusion-Test (komplett / nur Zentrum / nur Rand)
  4. Statistik-Test (Helligkeit / Kontrast / Schwarzanteil, Einzelmerkmal-AUC)
  5. Blur-Test      (Separation vs. Weichzeichnung)

Alle Abbildungen UND der komplette Textoutput landen in
    diagnostics_results/<label>/
sodass sich verschiedene Modellstaende (z.B. baseline vs. nach Rebalancing)
direkt nebeneinander vergleichen lassen.

Aufruf:
    python diagnostics.py --label baseline --checkpoint checkpoints/best_model.pth
    python diagnostics.py --label nach_norm --checkpoint checkpoints/best_model_v3.pth

Ohne --label wird ein Zeitstempel verwendet.
"""

import os
import argparse
from datetime import datetime

import numpy as np
import torch
from PIL import Image, ImageFilter
import matplotlib
matplotlib.use("Agg")               # kein Fenster, nur Dateien
import matplotlib.pyplot as plt

from torchvision import transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from data import transform, CLAHE, PerImageStandardize, get_data_loaders
from model.model import build_model
from evaluate import evaluate_model

# ---------------------------------------------------------------- Konfiguration
TEST_DIR = "data/chest_xray/test"
CLASSES = ["NORMAL", "PNEUMONIA"]
PNEUMONIA_IDX = 1
N_CAM_PER_CLASS = 3          # Bilder fuer die qualitativen CAM-Grids
N_QUANT_PER_CLASS = 100      # Bilder fuer Occlusion/Blur
N_STATS_PER_CLASS = 200      # Bilder fuer den Statistik-Test
BORDER_FRAC = 0.15
BLUR_RADII = [0, 2, 4, 8, 16]


# ---------------------------------------------------------------------- Helfer
class Report:
    """Sammelt Textoutput: gleichzeitig auf die Konsole und in eine Datei."""
    def __init__(self):
        self.lines = []

    def __call__(self, msg=""):
        print(msg)
        self.lines.append(msg)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


def sample_files(cls, n):
    folder = os.path.join(TEST_DIR, cls)
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))[:n]]


def load_tensor(path):
    img = Image.open(path).convert("RGB")
    return img, transform(img).unsqueeze(0)


@torch.no_grad()
def pneu_prob(model, tensor):
    return torch.softmax(model(tensor), dim=1)[0, PNEUMONIA_IDX].item()


# Bausteine der Pipeline einzeln, um den Blur an der KORREKTEN Stelle einzufuegen
# (nach CLAHE). Reihenfolge identisch zu data.transform.
_resize = transforms.Resize((224, 224))
_clahe = CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
_to_tensor = transforms.ToTensor()
_standardize = PerImageStandardize()


def blurred_input(pil_img, radius):
    """Modell-Input mit Blur an der korrekten Pipeline-Stelle:
    Resize -> CLAHE -> Blur -> ToTensor -> Standardize.
    So degradiert der Blur wirklich das, was das Modell sieht - CLAHE kann ihn
    nicht mehr aufschaerfen. radius=0 ist identisch zu data.transform."""
    x = _clahe(_resize(pil_img))
    if radius > 0:
        x = x.filter(ImageFilter.GaussianBlur(radius))
    return _standardize(_to_tensor(x)).unsqueeze(0)


# ------------------------------------------------------------ Test 0: Evaluation
def test_evaluation(model, say):
    say("\n=== Test 0: Klassifikations-Metriken auf dem Test-Set ===")
    say("Quantitative Leistung (Konfusionsmatrix, Sensitivitaet/Spezifitaet, AUC,")
    say("Youden-Schwellenwert) - damit De-Bias-Effekte auch an harten Zahlen sichtbar werden.")
    _, _, test_loader, classes = get_data_loaders()
    evaluate_model(model, test_loader, classes, say=say)


# ----------------------------------------------------------------- Test 1: CAM
def test_cam_methods(model, outdir, say):
    say("\n=== Test 1: CAM-Vergleich (Grad-CAM / Grad-CAM++ / HiResCAM) ===")
    say("Alle Heatmaps erklaeren die PNEUMONIA-Klasse. Grad-CAM==HiResCAM ist bei")
    say("ResNet (GAP->fc) erwartbar und belegt, dass das CAM 'faithful' ist.")
    target_layer = model.layer4[-1]
    methods = {
        "Grad-CAM": GradCAM(model=model, target_layers=[target_layer]),
        "Grad-CAM++": GradCAMPlusPlus(model=model, target_layers=[target_layer]),
        "HiResCAM": HiResCAM(model=model, target_layers=[target_layer]),
    }
    targets = [ClassifierOutputTarget(PNEUMONIA_IDX)]
    _cam_grid(model, methods, targets, outdir, "cam_compare.png")
    say("  -> gespeichert: cam_compare.png")


# --------------------------------------------------------------- Test 2: Layer
def test_cam_layers(model, outdir, say):
    say("\n=== Test 2: Layer-Vergleich (layer4 7x7 / layer3 14x14 / layer2 28x28) ===")
    say("Grad-CAM auf verschiedenen Bloecken. Tiefer = grober, aber semantischer.")
    say("HiResCAM hier bewusst nicht: seine Faithfulness gilt nur vor dem GAP.")
    methods = {
        "layer4 (7x7)": GradCAM(model=model, target_layers=[model.layer4[-1]]),
        "layer3 (14x14)": GradCAM(model=model, target_layers=[model.layer3[-1]]),
        "layer2 (28x28)": GradCAM(model=model, target_layers=[model.layer2[-1]]),
    }
    targets = [ClassifierOutputTarget(PNEUMONIA_IDX)]
    _cam_grid(model, methods, targets, outdir, "cam_layers.png")
    say("  -> gespeichert: cam_layers.png")


def _cam_grid(model, methods, targets, outdir, filename):
    n_cols = 1 + len(methods)
    n_rows = len(CLASSES) * N_CAM_PER_CLASS
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    row = 0
    for cls in CLASSES:
        for path in sample_files(cls, N_CAM_PER_CLASS):
            img, t = load_tensor(path)
            with torch.no_grad():
                probs = torch.softmax(model(t), dim=1)[0]
            pred = CLASSES[probs.argmax().item()]
            rgb = np.array(img.resize((224, 224))) / 255.0
            axes[row, 0].imshow(rgb)
            axes[row, 0].set_title(f"Wahr: {cls}\nTipp: {pred} "
                                   f"(P={probs[PNEUMONIA_IDX]:.2f})", fontsize=9)
            axes[row, 0].axis("off")
            for col, (name, cam) in enumerate(methods.items(), start=1):
                gcam = cam(input_tensor=t, targets=targets)[0]
                vis = show_cam_on_image(rgb, gcam, use_rgb=True)
                axes[row, col].imshow(vis)
                axes[row, col].set_title(name, fontsize=10)
                axes[row, col].axis("off")
            row += 1
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, filename), dpi=120)
    plt.close(fig)


# ----------------------------------------------------------- Test 3: Occlusion
def test_occlusion(model, outdir, say):
    say("\n=== Test 3: Occlusion (komplett / nur Zentrum / nur Rand) ===")
    say("Welcher Bildteil traegt die Vorhersage? Kollabiert die Separation bei")
    say("'nur Zentrum' UND 'nur Rand', ist das Signal weder rein zentral noch")
    say("rein am Rand lokalisiert - Hinweis auf globale/grobe Merkmale.")

    def mask_center_only(t):
        b = int(t.shape[-1] * BORDER_FRAC); o = t.clone()
        o[..., :b, :] = 0; o[..., -b:, :] = 0; o[..., :, :b] = 0; o[..., :, -b:] = 0
        return o

    def mask_border_only(t):
        b = int(t.shape[-1] * BORDER_FRAC); o = t.clone()
        o[..., b:-b, b:-b] = 0
        return o

    variants = {"komplett": lambda t: t,
                "nur Zentrum": mask_center_only,
                "nur Rand": mask_border_only}
    res = {v: {c: [] for c in CLASSES} for v in variants}
    for cls in CLASSES:
        for path in sample_files(cls, N_QUANT_PER_CLASS):
            _, t = load_tensor(path)
            for vname, fn in variants.items():
                res[vname][cls].append(pneu_prob(model, fn(t)))

    say(f"\n{'Variante':<14}{'P(PNEU|NORMAL)':>16}{'P(PNEU|PNEU)':>16}{'Separation':>14}")
    say("-" * 60)
    for vname in variants:
        mn = np.mean(res[vname]["NORMAL"]); mp = np.mean(res[vname]["PNEUMONIA"])
        say(f"{vname:<14}{mn:>16.3f}{mp:>16.3f}{mp - mn:>14.3f}")
    say("-" * 60)


# ------------------------------------------------------------ Test 4: Statistik
def test_stats(model, outdir, say):
    say("\n=== Test 4: Statistik-Test (Einzelmerkmal-Trennkraft) ===")
    say("Wie gut trennt EINE globale Zahl die Klassen? AUC nahe 0/1 = starker")
    say("Datensatz-Confounder. Merkmal ist bei AUC<0.5 in Pneumonie NIEDRIGER.")

    def features(img):
        g = np.array(img.convert("L").resize((224, 224))) / 255.0
        return {"mean_intensity": g.mean(), "contrast": g.std(),
                "black_frac": (g < 0.05).mean()}

    def auc(pos, neg):
        allv = np.concatenate([pos, neg])
        ranks = allv.argsort().argsort() + 1
        u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
        return u / (len(pos) * len(neg))

    names = ["mean_intensity", "contrast", "black_frac"]
    data = {c: {f: [] for f in names} for c in CLASSES}
    for cls in CLASSES:
        for path in sample_files(cls, N_STATS_PER_CLASS):
            fv = features(Image.open(path))
            for f in names:
                data[cls][f].append(fv[f])

    say(f"\n{'Merkmal':<16}{'mean NORMAL':>14}{'mean PNEU':>14}{'AUC':>8}{'Trennkraft':>12}")
    say("-" * 64)
    for f in names:
        mn = np.mean(data["NORMAL"][f]); mp = np.mean(data["PNEUMONIA"][f])
        a = auc(np.array(data["PNEUMONIA"][f]), np.array(data["NORMAL"][f]))
        say(f"{f:<16}{mn:>14.3f}{mp:>14.3f}{a:>8.3f}{abs(a - 0.5) * 2:>12.3f}")
    say("-" * 64)

    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4))
    for ax, f in zip(axes, names):
        ax.hist(data["NORMAL"][f], bins=30, alpha=0.6, label="NORMAL", density=True)
        ax.hist(data["PNEUMONIA"][f], bins=30, alpha=0.6, label="PNEUMONIA", density=True)
        ax.set_title(f); ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "bias_stats.png"), dpi=120)
    plt.close(fig)
    say("  -> gespeichert: bias_stats.png")


# ---------------------------------------------------------------- Test 5: Blur
def test_blur(model, outdir, say):
    say("\n=== Test 5: Blur-Test (Separation vs. Weichzeichnung) ===")
    say("Starker Blur zerstoert feine Anatomie, laesst Grobstatistik intakt.")
    say("Bleibt die Separation hoch, reicht dem Modell die Grobstruktur.")
    say("Blur wird NACH CLAHE angewandt (sonst schaerft CLAHE ihn wieder auf).")
    res = {r: {c: [] for c in CLASSES} for r in BLUR_RADII}
    for cls in CLASSES:
        for path in sample_files(cls, N_QUANT_PER_CLASS):
            img = Image.open(path).convert("RGB")
            for r in BLUR_RADII:
                res[r][cls].append(pneu_prob(model, blurred_input(img, r)))

    say(f"\n{'Blur-Radius':<12}{'P(PNEU|NORMAL)':>16}{'P(PNEU|PNEU)':>16}{'Separation':>14}")
    say("-" * 58)
    seps = []
    for r in BLUR_RADII:
        mn = np.mean(res[r]["NORMAL"]); mp = np.mean(res[r]["PNEUMONIA"])
        seps.append(mp - mn)
        say(f"{r:<12}{mn:>16.3f}{mp:>16.3f}{mp - mn:>14.3f}")
    say("-" * 58)

    fig = plt.figure(figsize=(7, 5))
    plt.plot(BLUR_RADII, seps, marker="o")
    plt.xlabel("Gauss-Blur-Radius (px)")
    plt.ylabel("Separation  P(PNEU|Pneu) - P(PNEU|Normal)")
    plt.title("Trennkraft vs. Weichzeichnung")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "blur_test.png"), dpi=120)
    plt.close(fig)
    say("  -> gespeichert: blur_test.png")


# ---------------------------------------------------------------------- Runner
def main():
    parser = argparse.ArgumentParser(description="Modell-Diagnostik, gebuendelt.")
    parser.add_argument("--label", default=None,
                        help="Ordnername unter diagnostics_results/ (Default: Zeitstempel)")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth",
                        help="Pfad zum Modell-Checkpoint")
    args = parser.parse_args()

    label = args.label or datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("diagnostics_results", label)
    os.makedirs(outdir, exist_ok=True)

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    say = Report()
    say("#" * 64)
    say(f"# Diagnostik-Lauf: {label}")
    say(f"# Checkpoint: {args.checkpoint}")
    say(f"# Datum: {datetime.now():%Y-%m-%d %H:%M:%S}")
    say("#" * 64)

    test_evaluation(model, say)
    test_cam_methods(model, outdir, say)
    test_cam_layers(model, outdir, say)
    test_occlusion(model, outdir, say)
    test_stats(model, outdir, say)
    test_blur(model, outdir, say)

    report_path = os.path.join(outdir, "report.txt")
    say.save(report_path)
    print(f"\nAlle Ergebnisse gespeichert in: {outdir}")
    print(f"Textreport: {report_path}")


if __name__ == "__main__":
    main()
