"""
Erzeugt Splits, die fuer ALLE drei Varianten identisch sind.

Zwei Dinge sind hier wichtig:

1. Gruppierung nach Patient. Im Kermany-Datensatz heissen die Dateien
   person17_bacteria_43.jpeg -- ein Patient hat mehrere Aufnahmen. Bei
   zufaelligem Split landen Bilder desselben Patienten in Train UND Val,
   und die Metriken sind zu optimistisch. StratifiedGroupKFold verhindert das.

2. Der offizielle test/-Ordner bleibt unangetastet und wird erst ganz am Ende
   einmal ausgewertet. Verglichen werden die Varianten auf den CV-Folds von
   train+val -- das gibt mehr statistische Power als ein einzelner Split mit
   624 Testbildern.

CLI:
  python splits.py --root data/prepared/crop --out splits.json --folds 5
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

CLASS_NAMES = {"normal": 0, "pneumonia": 1}
SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_record(rel: Path) -> dict | None:
    """Leitet Klasse, Original-Split und Patienten-Gruppe aus dem Pfad ab."""
    parts = [p.lower() for p in rel.parts]
    cls = next((p for p in parts if p in CLASS_NAMES), None)
    if cls is None:
        return None
    split = next((p for p in parts if p in SPLIT_NAMES), "all")
    if split in {"valid", "validation"}:
        split = "val"

    name = rel.name
    m = re.search(r"person[_\-]?(\d+)", name, re.I)
    if m:
        # Patienten-IDs sind pro Original-Split und Klasse neu vergeben,
        # deshalb muessen beide in den Schluessel
        group = f"{split}|{cls}|person{m.group(1)}"
    else:
        m = re.search(r"IM-?(\d+)", name, re.I)
        group = f"{split}|{cls}|im{m.group(1)}" if m else f"{split}|{cls}|{rel.stem}"

    return {"file": str(rel), "label": CLASS_NAMES[cls], "split": split, "group": group}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path,
                   help="Ordner EINER Variante, z.B. data/prepared/crop")
    p.add_argument("--out", type=Path, default=Path("splits.json"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    records = []
    for f in sorted(x for x in args.root.rglob("*") if x.suffix.lower() in EXTS):
        rec = parse_record(f.relative_to(args.root))
        if rec:
            records.append(rec)
    if not records:
        raise SystemExit(f"Keine Bilder mit erkennbarer Klasse unter {args.root}")

    dev = [r for r in records if r["split"] != "test"]
    test = [r for r in records if r["split"] == "test"]
    if not dev:  # keine Split-Ordner vorhanden -> alles ist Entwicklungsmenge
        dev, test = records, []

    y = np.array([r["label"] for r in dev])
    g = np.array([r["group"] for r in dev])
    n_groups = len(set(g))
    print(f"Entwicklungsmenge: {len(dev)} Bilder, {n_groups} Patientengruppen")
    print(f"  Klassen: {Counter(r['label'] for r in dev)}")
    print(f"Holdout (offizielles test/): {len(test)} Bilder")
    if n_groups < args.folds * 2:
        print("  WARNUNG: sehr wenige Gruppen -- Gruppierung greift kaum.")

    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = []
    for k, (tr, va) in enumerate(sgkf.split(np.zeros(len(dev)), y, g)):
        assert not (set(g[tr]) & set(g[va])), "Gruppen-Leak!"
        folds.append({"train": [dev[i]["file"] for i in tr],
                      "val": [dev[i]["file"] for i in va]})
        print(f"  Fold {k}: train {len(tr)}, val {len(va)}, "
              f"val-Positivrate {y[va].mean():.3f}")

    payload = {
        "labels": {r["file"]: r["label"] for r in records},
        "folds": folds,
        "holdout": [r["file"] for r in test],
    }
    args.out.write_text(json.dumps(payload))
    print(f"\ngespeichert: {args.out}")


if __name__ == "__main__":
    main()
