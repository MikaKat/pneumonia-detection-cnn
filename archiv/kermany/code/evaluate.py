"""Bewertet ein Modell auf dem Test-Set: Konfusionsmatrix, Sensitivität/Spezifität,
AUC, Classification-Report und Youden-optimierter Schwellenwert.

Als Skript ausführbar (lädt checkpoints/best_model_v3.pth). Die Kernfunktion
evaluate_model() wird zusätzlich von diagnostics.py als "Test 0" wiederverwendet,
damit die quantitative Leistung im selben Report neben den Bias-Tests steht.
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from tqdm import tqdm

from data import get_data_loaders
from model.model import build_model

CHECKPOINT_PATH = "checkpoints/best_model_v3.pth"


def evaluate_model(model, test_loader, classes, say=print):
    """Rechnet die Test-Metriken und gibt sie über `say` aus (print ODER ein
    Report-Objekt, das zugleich in eine Datei schreibt). Erwartet ein bereits
    geladenes, mit .eval() gesetztes Modell und den passenden test_loader."""
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test Evaluation"):
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
            all_probs.extend(probs[:, 1].numpy())    # Index 1 = PNEUMONIA
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    say(f"Konfusionsmatrix (Zeilen=wahr, Spalten=Vorhersage):\n{cm}")
    say(f"Sensitivität (Recall PNEU): {tp / (tp + fn):.4f}")
    say(f"Spezifität:                 {tn / (tn + fp):.4f}")
    say(f"AUC:                        {roc_auc_score(all_labels, all_probs):.4f}")
    say("\n" + classification_report(all_labels, all_preds, target_names=classes))

    # Schwellenwert optimieren (Youden's J = maximiert Sensitivität + Spezifität - 1)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]
    say(f"Bester Schwellenwert (Youden J): {best_threshold:.4f}")
    say(f"  -> Sensitivität: {tpr[best_idx]:.4f}, Spezifität: {1 - fpr[best_idx]:.4f}")
    new_cm = confusion_matrix(all_labels, (all_probs >= best_threshold).astype(int))
    say(f"Konfusionsmatrix bei diesem Schwellenwert:\n{new_cm}")


def evaluate():
    _, _, test_loader, classes = get_data_loaders()
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()
    evaluate_model(model, test_loader, classes)


if __name__ == "__main__":
    evaluate()
