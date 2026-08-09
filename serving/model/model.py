"""Modell-Definitionen fuer den Serving-Prozess.

Zwei Modelle stehen hier nebeneinander:

  `build_model`      ResNet18 mit auswechselbarem Kopf. Die alte Kermany-Strecke
                     (2 Klassen) und der einkoepfige RSNA-Klassifikator
                     (1 Logit) haben damit geladen. Bleibt erhalten, weil die
                     Tests und die frueheren Gewichte darauf zeigen.

  `TwoHeadNet`       Das ausgelieferte Modell der Phase 10. ResNet18 mit einem
                     ZWEITEN Ausgang: einem 14x14-Feld, das sagt WO. Fuenf
                     Gewichte dieser Bauart bilden das Ensemble, das die App
                     rechnet.

Warum die Klasse hier NACHGEBAUT und nicht importiert wird
----------------------------------------------------------
Das Original steht in `rsna/pipeline/rsna_train.py`. Dieses Modul mitzuziehen
haette den Serving-Prozess an die halbe Trainingsstrecke gebunden: pandas,
sklearn, den Argument-Parser, die Datensatzklassen und die Augmentierung. Der
Container soll klein bleiben und beim Start nichts laden, was er zum Rechnen
eines einzelnen Bildes nicht braucht. Dieselbe Abwaegung ist schon einmal so
ausgegangen, naemlich bei der Vorverarbeitung in `main.py`.

Der Preis ist bekannt und wird nicht verschwiegen: es gibt die Klasse jetzt
zweimal, und zwei Fassungen koennen auseinanderlaufen. Dagegen steht
`tests/test_serving_ensemble.py`. Der Test baut BEIDE Fassungen und vergleicht
die Namen und Formen aller Parameter sowie die Ausgabeformen. Laeuft eine der
beiden weg, faellt der Test, und zwar bevor ein Gewicht still falsch geladen
wird. Wer hier etwas aendert, aendert es dort mit.

Quelle des Nachbaus: `rsna/pipeline/rsna_train.py`, Klassen `TwoHeadNet` und
`ClassifierView`, Konstante `HEAD_GRID`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torchvision import models

NUM_CLASSES = 2

# Gemessen und nicht gewaehlt: rsna/befunde/rsna_kopfraster.py. Ein feineres
# Raster lohnt nicht, die Decke des 14x14-Rasters auf dem Wettbewerbsmass liegt
# bei 0,8111. Die Zahl muss mit rsna_train.HEAD_GRID uebereinstimmen, sonst
# passen die Gewichte nicht.
HEAD_GRID = 14


def build_model(pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Baut ResNet18 auf.

    pretrained=True: lädt ImageNet-Gewichte und friert alle Schichten außer der
    letzten ein (Transfer Learning) - für den Trainingsstart.

    pretrained=False: erzeugt nur die Architektur ohne neue ImageNet-Gewichte -
    dafür gedacht, direkt danach eigene trainierte Gewichte hineinzuladen
    (siehe evaluate.py), ohne unnötig ImageNet-Gewichte herunterzuladen.

    num_classes: Anzahl Ausgänge der letzten Schicht. Vorgabe 2 (Kermany-Strecke,
    CrossEntropy über NORMAL/PNEUMONIA). Die RSNA-Strecke trainiert mit EINEM
    Logit und BCE (rsna/pipeline/rsna_train.py: `nn.Linear(in_features, 1)`),
    deshalb muss der Kopf hier einstellbar sein - sonst scheitert
    `load_state_dict` an der Form von `fc.weight`.
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.resnet18(weights=weights)

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


class TwoHeadNet(nn.Module):
    """ResNet18 mit einem zweiten Ausgang: einem grid x grid grossen Feld.

    Der Klassifikationsweg ist die gewoehnliche `resnet18.forward`, Schritt fuer
    Schritt ausgeschrieben statt nachgebaut, damit er genau das rechnet, was das
    einkoepfige Modell rechnet.

    Der Kopf zapft `layer3` an. Bei 224 Bildpunkten ist layer3 bereits 14 x 14,
    also genau das Raster; von `layer4` aus dorthin zu kommen haette dessen
    Schrittweite geaendert und damit auch den Klassifikationsweg. Das
    `adaptive_avg_pool2d` auf ein FESTES Raster sorgt ausserdem dafuer, dass der
    Kopf sein 14 x 14 behaelt, egal wie gross das Eingabebild ist.

    Der Kopf ist eine einzige 1x1-Faltung, also eine logistische Regression je
    Kachel auf den 256 layer3-Kanaelen.

    ACHTUNG, die Namen sind Teil des Vertrags: die Gewichte in den Dateien
    `checkpoints/rsna_f{0..4}_s0_p5head_ex.pth` heissen `trunk.*` und `loc.*`.
    Wer hier ein Attribut umbenennt, macht die fuenf Dateien unladbar.
    """

    def __init__(self, grid: int = HEAD_GRID, pretrained: bool = False):
        super().__init__()
        # Vorgabe `pretrained=False`, anders als im Trainingsmodul: hier wird
        # ausschliesslich geladen, nie trainiert. Ein ImageNet-Download beim
        # Start des Containers waere 45 MB fuer Gewichte, die eine Zeile
        # spaeter ueberschrieben werden.
        m = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 1)
        self.trunk = m
        self.grid = grid
        self.loc = nn.Conv2d(256, 1, kernel_size=1)
        nn.init.normal_(self.loc.weight, std=0.01)
        nn.init.zeros_(self.loc.bias)

    def features3(self, x):
        m = self.trunk
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        return m.layer3(m.layer2(m.layer1(x)))

    def forward(self, x):
        f3 = self.features3(x)
        f4 = self.trunk.layer4(f3)
        logit = self.trunk.fc(torch.flatten(self.trunk.avgpool(f4), 1))
        field = self.loc(Fn.adaptive_avg_pool2d(f3, self.grid))
        return logit, field


class ClassifierView(nn.Module):
    """Nur der Klassifikationszweig, damit Grad-CAM einen schlichten
    Klassifikator sieht.

    `pytorch_grad_cam` erwartet ein Modul, das EINEN Tensor zurueckgibt, und es
    haengt sich an eine Schicht ueber deren Objektidentitaet. Ein zweikoepfiges
    Modell gibt ein Tupel zurueck und bricht daran. Diese Huelle reicht den
    Logit heraus und stellt `layer4` desselben Objekts nach aussen. Deshalb
    zeigt `GradCAM(model=view, target_layers=[view.layer4])` auf genau den
    Block, auf den es beim einkoepfigen Modell auch gezeigt hat.

    Das ist die Antwort auf die Frage aus der Arbeitsliste, wie Grad-CAM beim
    zweikoepfigen Modell neu zu zielen ist: gar nicht auf eine andere Schicht,
    sondern auf dieselbe Schicht durch eine Huelle hindurch.
    """

    def __init__(self, net: TwoHeadNet):
        super().__init__()
        self.net = net
        self.layer4 = net.trunk.layer4

    def forward(self, x):
        return self.net(x)[0]


def build_two_head_model(grid: int = HEAD_GRID) -> TwoHeadNet:
    """Die leere Architektur der Phase-10-Gewichte, im Auswertungsmodus."""
    model = TwoHeadNet(grid=grid, pretrained=False)
    model.eval()
    return model
