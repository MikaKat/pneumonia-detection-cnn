"""
Produces splits that are identical for ALL three variants.

Two things matter here:

1. Grouping by patient. In the Kermany dataset the files are named
   person17_bacteria_43.jpeg, and one patient has several acquisitions. With a
   random split, images of the same patient end up in train AND val, and the
   metrics come out too optimistic. StratifiedGroupKFold prevents that. It cuts
   the data into folds, each serving once as the validation set while the model
   trains on the rest, and it never spreads one patient's images over two
   folds. The class ratio also stays steady from fold to fold.

2. The official test/ folder stays untouched and is evaluated exactly once, at
   the very end. The variants are compared on the CV folds of train+val, which
   gives more statistical power than a single split with 624 test images.

Interpreting the output:
  The run first prints the size of the development set, the number of patient
  groups and the class counts, then the size of the holdout, i.e. the official
  test/ folder. Per fold it reports the train and val sizes and the positive
  rate in val; the reference is the overall class balance printed above, and
  the per-fold positive rates should stay close to it. If they drift, fold
  results are not comparable. Far fewer groups than images means the patient
  grouping is doing work; if the group count approaches the image count, the
  filenames carry no patient information and the guarantee is void.
  The warning about very few groups and the group-leak assert are the failure
  conditions: the assert aborts the run, and any leak makes the split unusable.
  The written splits.json is what all later runs consume, so every variant is
  evaluated on exactly the same partition.

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
    """Derives class, original split and patient group from the file path."""
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
        # patient IDs are assigned anew per original split and class,
        # so both have to go into the key
        group = f"{split}|{cls}|person{m.group(1)}"
    else:
        m = re.search(r"IM-?(\d+)", name, re.I)
        group = f"{split}|{cls}|im{m.group(1)}" if m else f"{split}|{cls}|{rel.stem}"

    return {"file": str(rel), "label": CLASS_NAMES[cls], "split": split, "group": group}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path,
                   help="folder of ONE variant, e.g. data/prepared/crop")
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
        raise SystemExit(f"No images with a recognisable class under {args.root}")

    dev = [r for r in records if r["split"] != "test"]
    test = [r for r in records if r["split"] == "test"]
    if not dev:  # no split folders present -> everything is the development set
        dev, test = records, []

    y = np.array([r["label"] for r in dev])
    g = np.array([r["group"] for r in dev])
    n_groups = len(set(g))
    print(f"Development set: {len(dev)} images, {n_groups} patient groups")
    print(f"  Classes: {Counter(r['label'] for r in dev)}")
    print(f"Holdout (official test/): {len(test)} images")
    if n_groups < args.folds * 2:
        print("  WARNING: very few groups, the grouping hardly takes effect.")

    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = []
    for k, (tr, va) in enumerate(sgkf.split(np.zeros(len(dev)), y, g)):
        assert not (set(g[tr]) & set(g[va])), "group leak!"
        folds.append({"train": [dev[i]["file"] for i in tr],
                      "val": [dev[i]["file"] for i in va]})
        print(f"  Fold {k}: train {len(tr)}, val {len(va)}, "
              f"val positive rate {y[va].mean():.3f}")

    payload = {
        "labels": {r["file"]: r["label"] for r in records},
        "folds": folds,
        "holdout": [r["file"] for r in test],
    }
    args.out.write_text(json.dumps(payload))
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
