"""
Splits for RSNA, stratified by label AND projection.

Three differences to `splits.py` (Kermany), each one born from a concrete
mistake in the previous phase:

1. Stratification is by `label x ViewPosition`, not by label alone.
   Stratifying means every part of the split gets the same mix of cases as the
   full table. The confounder check showed that `ViewPosition` on its own
   separates the classes at AUC 0.706, the probability that a random pneumonia
   case is ranked above a random non-case (the same quantity as a c-statistic).
   That is practically the ENTIRE metadata leak: within a single projection,
   0.553 remains, i.e. noise. If the AP/PA ratio varies between the folds, the
   fold AUCs vary for a reason that has nothing to do with the model. The later
   per-stratum evaluation also needs enough cases of both classes in EVERY
   fold.

2. What is stored are `patientId` strings, not file paths.
   On Kermany, `splits.json` contained Windows backslashes; `parse_record` then
   silently returned `None` instead of failing loudly, and the inner selection
   split was broken for a while without anyone noticing. IDs are
   platform-neutral, and the dataset loader assembles the path from root + ID +
   suffix.

3. A real holdout is separated off. The official `stage_2_test_images` folder
   has NO labels (that was the competition test set), so it is no substitute.
   15 % of the labelled images are therefore sealed away before anything else
   happens and are not touched until the very end. The remaining 85 % are cut
   into 5 folds, each of which serves once as the validation set while the
   model trains on the other four.

Grouping by `patientId`: in RSNA there is one image per patient, so the
grouping is effectively inert. It stays in anyway. It costs nothing, and the
guarantee "no group in both train and val" is then checked rather than assumed.

Interpreting the output:
  For the holdout, the dev set and every validation fold, one line reports the
  number of images, the positive rate and, per projection, its share of the
  subset together with the positive rate inside it. The reference value is the
  positive rate of the full table, printed in brackets as the target; subset
  values should sit close to it, and the projection shares should be alike
  across folds. Large deviations mean the stratification did not take, and the
  fold AUCs are then not comparable. The asserts are the hard criteria: any
  group overlap between train and val, any overlap with the holdout, or
  validation folds that do not cover the dev set will abort the run. The final
  "smallest cell" number is the minimum count of one projection x class
  combination in a validation fold; below 50 a per-stratum AUC carries no
  weight, and the printed warning says so.

CLI:
  python rsna_splits.py --csv data/rsna --headers qc/rsna/dicom_headers.csv \
      --out rsna_splits.json --folds 5 --holdout 0.15
"""

from __future__ import annotations

import _repo_path  # noqa: F401  (setzt sys.path fuer die Nachbarordner)

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from rsna_data import load_labels, scan_headers


def build_table(csv_dir: Path, headers: Path | None, dicom_dir: Path | None,
                mode: str) -> pd.DataFrame:
    """Labels + ViewPosition in a single table."""
    lab = load_labels(csv_dir, mode=mode)
    if headers and Path(headers).exists():
        hdr = scan_headers(dicom_dir or Path("."), cache=headers)
    elif dicom_dir:
        hdr = scan_headers(dicom_dir, cache=headers)
    else:
        raise SystemExit("Neither --headers nor --dicom was given.")

    df = lab.merge(hdr[["patientId", "ViewPosition"]], on="patientId", how="inner")
    if len(df) < len(lab):
        print(f"WARNING: {len(lab) - len(df)} images without header, dropped.")
    df["ViewPosition"] = df["ViewPosition"].fillna("UNKNOWN").astype(str)
    # The stratification key: label and projection jointly.
    df["stratum"] = df["label"].astype(str) + "|" + df["ViewPosition"]
    return df


def report(name: str, sub: pd.DataFrame, total: pd.DataFrame) -> None:
    """Compares the composition of a subset with that of the full table."""
    line = (f"  {name:<12}n={len(sub):>6}  pos={sub['label'].mean():.3f}"
            f"  (target {total['label'].mean():.3f})")
    for vp in sorted(total["ViewPosition"].unique()):
        s = sub[sub["ViewPosition"] == vp]
        line += f"   {vp} {len(s) / max(len(sub), 1):.3f}/pos {s['label'].mean():.3f}"
    print(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--headers", type=Path, default=Path("qc/rsna/dicom_headers.csv"),
                   help="header cache from rsna_metadata_leak_check.py")
    p.add_argument("--dicom", type=Path, default=None,
                   help="only needed if the header cache is missing")
    p.add_argument("--out", type=Path, default=Path("rsna_splits.json"))
    p.add_argument("--mode", default="clinical", choices=["clinical", "strict"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--holdout", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = build_table(args.csv, args.headers, args.dicom, args.mode)
    print(f"\n{len(df)} images, mode '{args.mode}', "
          f"{df['patientId'].nunique()} patient groups")
    print("Strata:", dict(sorted(Counter(df["stratum"]).items())))

    ids = df["patientId"].values
    y = df["label"].values
    strat = df["stratum"].values
    groups = df["patientId"].values          # = ids, see module docstring

    # ---- 1. Separate off the holdout ---------------------------------------
    # StratifiedGroupKFold instead of train_test_split: the same mechanism as
    # for the CV folds, hence the same guarantees. n_splits sets the fraction.
    n_out = max(2, round(1 / args.holdout))
    sgkf = StratifiedGroupKFold(n_splits=n_out, shuffle=True, random_state=args.seed)
    dev_idx, hold_idx = next(iter(sgkf.split(np.zeros(len(df)), strat, groups)))

    dev, hold = df.iloc[dev_idx], df.iloc[hold_idx]
    assert not (set(dev["patientId"]) & set(hold["patientId"]))
    print(f"\nHoldout (untouched until the very end, ~1/{n_out}):")
    report("holdout", hold, df)
    report("dev", dev, df)

    # ---- 2. CV folds on the development set --------------------------------
    print(f"\n{args.folds} folds on the development set:")
    sgkf2 = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                 random_state=args.seed)
    folds, seen_val = [], set()
    for k, (tr, va) in enumerate(sgkf2.split(np.zeros(len(dev)),
                                             dev["stratum"].values,
                                             dev["patientId"].values)):
        tr_ids = dev.iloc[tr]["patientId"].tolist()
        va_ids = dev.iloc[va]["patientId"].tolist()
        assert not (set(tr_ids) & set(va_ids)), f"group leak in fold {k}"
        assert not (set(va_ids) & set(hold["patientId"])), f"holdout leak in fold {k}"
        seen_val |= set(va_ids)
        folds.append({"train": tr_ids, "val": va_ids})
        report(f"fold {k} val", dev.iloc[va], df)

    assert seen_val == set(dev["patientId"]), "val folds do not cover the dev set"

    # Smallest cell: decides whether a per-projection evaluation holds at all.
    worst = min(
        len(dev.iloc[va][(dev.iloc[va]["ViewPosition"] == vp) &
                         (dev.iloc[va]["label"] == lb)])
        for _, va in sgkf2.split(np.zeros(len(dev)), dev["stratum"].values,
                                 dev["patientId"].values)
        for vp in df["ViewPosition"].unique() for lb in (0, 1))
    print(f"\nSmallest cell (fold x projection x class): {worst} cases")
    if worst < 50:
        print("  WARNING: too few for a dependable per-stratum AUC.")

    payload = {
        "meta": {"dataset": "rsna", "mode": args.mode, "folds": args.folds,
                 "holdout_frac": args.holdout, "seed": args.seed,
                 "n_images": int(len(df)), "key": "patientId"},
        "labels": {str(i): int(v) for i, v in zip(ids, y)},
        "viewpos": dict(zip(df["patientId"], df["ViewPosition"])),
        "folds": folds,
        "holdout": hold["patientId"].tolist(),
    }
    args.out.write_text(json.dumps(payload))
    print(f"\nsaved: {args.out}")
    print("Keys are patientId strings, not paths. The loader assembles")
    print("the path as  <root>/<patientId>.png")


if __name__ == "__main__":
    main()
