"""
Checks a finished rsna_splits.json against the guarantees it is meant to give.

The assertions in `rsna_splits.py` run while the file is being written. This
one runs on the written file, which is what the training actually reads. On
Kermany the bug sat in exactly that gap: the file held Windows backslashes,
the parser returned `None` instead of complaining, and the inner selection
split stayed broken for weeks while every assertion kept passing.

  python test_rsna_splits.py --splits rsna_splits.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

FAILED = []


def check(name: str, cond: bool, info: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FAILED.append(name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    p.add_argument("--images", type=Path, default=None,
                   help="optional: PNG folder, checks that every ID has a file")
    p.add_argument("--min-cell", type=int, default=50)
    args = p.parse_args()

    d = json.loads(args.splits.read_text())
    labels, viewpos, folds, hold = (d["labels"], d["viewpos"],
                                    d["folds"], set(d["holdout"]))
    all_ids = set(labels)
    print(f"\n{args.splits}: {len(all_ids)} images, {len(folds)} folds, "
          f"{len(hold)} in the holdout, mode '{d['meta']['mode']}'")

    print("\nstructure")
    check("keys are IDs, not paths",
          not any(("/" in i or "\\" in i or i.endswith(".png")) for i in all_ids))
    check("viewpos covers all IDs", set(viewpos) == all_ids)
    check("labels are 0/1", set(labels.values()) <= {0, 1})

    # An image in both train and val is scored by a model that has already
    # seen it, so the fold AUC no longer describes unseen cases.
    print("\nseparation of the subsets")
    dev_seen: Counter[str] = Counter()
    for k, f in enumerate(folds):
        tr, va = set(f["train"]), set(f["val"])
        check(f"fold {k}: no image in both train and val", not (tr & va), f"{len(tr & va)}")
        check(f"fold {k}: no holdout in train", not (tr & hold), f"{len(tr & hold)}")
        check(f"fold {k}: no holdout in val", not (va & hold), f"{len(va & hold)}")
        check(f"fold {k}: only known IDs", (tr | va) <= all_ids)
        dev_seen.update(va)

    dev = all_ids - hold
    check("val folds cover the dev set exactly", set(dev_seen) == dev,
          f"missing {len(dev - set(dev_seen))}, extra {len(set(dev_seen) - dev)}")
    check("every dev ID exactly once in val", set(dev_seen.values()) == {1},
          str(dict(Counter(dev_seen.values()))))

    # Stratification means every subset carries the same positive rate and the
    # same AP share as the full table. Where it fails, the folds differ in
    # composition, and their AUCs are no longer comparable with each other.
    print("\nstratification (target values from the full set)")
    pos_all = sum(labels.values()) / len(labels)
    ap_all = sum(v == "AP" for v in viewpos.values()) / len(viewpos)
    print(f"  overall: pos {pos_all:.3f}, AP share {ap_all:.3f}")
    for k, f in enumerate(folds):
        va = f["val"]
        pos = sum(labels[i] for i in va) / len(va)
        ap = sum(viewpos[i] == "AP" for i in va) / len(va)
        check(f"fold {k} val: positive rate near target", abs(pos - pos_all) < 0.02,
              f"{pos:.3f}")
        check(f"fold {k} val: AP share near target", abs(ap - ap_all) < 0.02,
              f"{ap:.3f}")
    hpos = sum(labels[i] for i in hold) / len(hold)
    hap = sum(viewpos[i] == "AP" for i in hold) / len(hold)
    check("holdout: positive rate near target", abs(hpos - pos_all) < 0.02, f"{hpos:.3f}")
    check("holdout: AP share near target", abs(hap - ap_all) < 0.02, f"{hap:.3f}")

    # Smallest cell: the fewest images any one projection and class combination
    # has in a validation fold. Where that number is small, an AUC computed
    # inside the stratum carries no weight.
    print("\ncell counts (fold x projection x class)")
    worst, where = 10**9, ""
    for k, f in enumerate(folds):
        for vp in ("AP", "PA"):
            for lb in (0, 1):
                n = sum(1 for i in f["val"] if viewpos[i] == vp and labels[i] == lb)
                if n < worst:
                    worst, where = n, f"fold {k} {vp} label={lb}"
    check(f"smallest cell >= {args.min_cell}", worst >= args.min_cell,
          f"{worst} ({where})")

    if args.images:
        print("\nfiles")
        missing = [i for i in all_ids if not (args.images / f"{i}.png").exists()]
        check("every ID has a PNG", not missing,
              f"{len(missing)} missing, e.g. {missing[:3]}")

    print("\n" + ("ALL TESTS PASSED" if not FAILED
                  else f"{len(FAILED)} FAILED: {FAILED}"))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
