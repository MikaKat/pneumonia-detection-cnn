"""Lädt das beste gespeicherte Modell (best_model.pth) und bewertet es auf dem Test-Set.

Schnell (Sekunden statt Minuten), da kein Training nötig ist - beliebig oft
wiederholbar/anpassbar, z.B. um mit Schwellenwerten zu experimentieren.
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from tqdm import tqdm

from data import get_data_loaders
from model import build_model

CHECKPOINT_PATH = "best_model.pth"


def evaluate():
    _, _, test_loader, classes = get_data_loaders()

    model = build_model(pretrained=False)  # Architektur ohne neue ImageNet-Gewichte
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test Evaluation"):
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
            all_probs.extend(probs[:, 1].numpy())  # Index 1 = PNEUMONIA

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    print("Konfusionsmatrix:\n", cm)

    tn, fp, fn, tp = cm.ravel()
    print(f"Sensitivität: {tp/(tp+fn):.4f}")
    print(f"Spezifität:  {tn/(tn+fp):.4f}")
    print(f"AUC:         {roc_auc_score(all_labels, all_probs):.4f}")
    print("\n", classification_report(all_labels, all_preds, target_names=classes))

    # Schwellenwert optimieren (Youden's J-Statistik: maximiert Sensitivität + Spezifität - 1)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    print(f"\nBester Schwellenwert: {best_threshold:.4f}")
    print(f"  -> Sensitivität: {tpr[best_idx]:.4f}, Spezifität: {1 - fpr[best_idx]:.4f}")

    new_preds = (all_probs >= best_threshold).astype(int)
    print("Neue Konfusionsmatrix:\n", confusion_matrix(all_labels, new_preds))


if __name__ == "__main__":
    evaluate()
