"""Modell-Definition: ResNet18 mit Transfer Learning für 2 Klassen (NORMAL, PNEUMONIA)."""

import torch.nn as nn
from torchvision import models

NUM_CLASSES = 2


def build_model(pretrained: bool = True) -> nn.Module:
    """Baut ResNet18 auf.

    pretrained=True: lädt ImageNet-Gewichte und friert alle Schichten außer der
    letzten ein (Transfer Learning) - für den Trainingsstart.

    pretrained=False: erzeugt nur die Architektur ohne neue ImageNet-Gewichte -
    dafür gedacht, direkt danach eigene trainierte Gewichte hineinzuladen
    (siehe evaluate.py), ohne unnötig ImageNet-Gewichte herunterzuladen.
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.resnet18(weights=weights)

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)

    return model
