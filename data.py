"""Datenladen für die Chest X-Ray Pneumonie-Klassifikation."""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

DATA_DIR = "data/chest_xray"
BATCH_SIZE = 32
VAL_SIZE = 100          # Anzahl Bilder, die vom Trainingsset für die Validierung abgezweigt werden
RANDOM_STATE = 42        # macht den Split reproduzierbar

# Transformationen: Bildgröße anpassen, in Tensor umwandeln, normalisieren
# (Werte entsprechen dem, worauf ResNet18 ursprünglich mit ImageNet trainiert wurde)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def get_data_loaders():
    """Lädt Trainings-, Validierungs- und Test-Daten und gibt die passenden
    DataLoader sowie die Klassennamen zurück."""
    train_data = datasets.ImageFolder(DATA_DIR + "/train", transform=transform)
    test_data = datasets.ImageFolder(DATA_DIR + "/test", transform=transform)

    # Der Val-Ordner des Original-Datensatzes hat nur 16 Bilder -> zu klein für eine
    # verlässliche Validierung. Stattdessen zweigen wir stratifiziert (gleiche
    # Klassen-Verteilung) einen Teil des Trainingssets ab.
    targets = train_data.targets
    train_idx, val_idx = train_test_split(
        list(range(len(train_data))),
        test_size=VAL_SIZE,
        stratify=targets,
        random_state=RANDOM_STATE
    )
    train_subset = Subset(train_data, train_idx)
    val_subset = Subset(train_data, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, train_data.classes


if __name__ == "__main__":
    # Datei direkt ausführen, um die Datengrößen zu prüfen (kein Training)
    train_loader, val_loader, test_loader, classes = get_data_loaders()
    print("Klassen:", classes)
    print("Anzahl Trainingsbilder:", len(train_loader.dataset))
    print("Anzahl Validierungsbilder:", len(val_loader.dataset))
    print("Anzahl Testbilder:", len(test_loader.dataset))
