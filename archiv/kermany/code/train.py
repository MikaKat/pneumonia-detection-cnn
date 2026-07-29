"""Trainiert das Modell und speichert die beste Version (nach Val Loss) als checkpoints/best_model_v3.pth.

Achtung: rechenintensiv (mehrere Minuten pro Epoche auf der CPU) - nur ausführen,
wenn wirklich neu trainiert werden soll. Für Auswertung/Experimente: evaluate.py nutzen,
das lädt nur das gespeicherte Modell und ist in Sekunden fertig.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import get_data_loaders
from model.model import build_model

NUM_EPOCHS = 5
LEARNING_RATE = 0.001
CHECKPOINT_PATH = "checkpoints/best_model_v3.pth"   # v3 = CLAHE + per-Bild-Norm + Rebalancing; v2/baseline bleiben


def train():
    train_loader, val_loader, _, classes = get_data_loaders()
    model = build_model(pretrained=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)  # nur die neue Schicht wird trainiert

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        # --- Training ---
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> neues bestes Modell gespeichert (Val Loss: {val_loss:.4f})")


if __name__ == "__main__":
    train()
