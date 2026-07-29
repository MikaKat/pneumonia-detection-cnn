"""Schritt 6 (Training): ResNet18 auf LUNGEN-MASKIERTER Eingabe trainieren -> v4.

Bewusst identisch zu train.py (5 Epochen, nur die fc-Schicht wird trainiert,
gleicher Sampler/Val-Split) - der EINZIGE Unterschied ist die maskierte Eingabe
aus data_masked.py. So ist die Differenz v3 -> v4 allein der Maskierung zuzuschreiben.

Aufruf:  python -m train_masked   (rechenintensiv - nur zum Neutrainieren)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_masked import get_masked_data_loaders
from model.model import build_model

NUM_EPOCHS = 5
LEARNING_RATE = 0.001
CHECKPOINT_PATH = "checkpoints/best_model_v4.pth"   # v4 = v3 + Lungenmaskierung


def train():
    train_loader, val_loader, _, classes = get_masked_data_loaders()
    model = build_model(pretrained=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)  # nur die neue Schicht

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
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

        model.eval()
        val_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
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
