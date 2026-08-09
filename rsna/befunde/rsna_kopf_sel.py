"""Head fields for the SELECTION split, so a threshold can be set honestly.

WHAT THIS PRODUCES
------------------
`head_sel_f{fold}_s{seed}.npz` beside the existing `head_f{fold}_s{seed}.npz`,
same format: patientId, field, grid. One forward pass of the saved weights over
the selection split. No training, no gradient, nothing is fitted here.

WHY IT EXISTS
-------------
Two open questions both need the head on data the model neither fitted on nor
reports on, and there is exactly one such split in this project: the inner
selection split.

  * THE THRESHOLD for phase 5b part 2. Turning a probability field into boxes
    needs a cut, and the roadmap says in writing that the cut comes from the
    selection split. Setting it on the validation split would report a number
    that was tuned on the same images, which is the one thing the whole split
    design exists to prevent. Only the validation fields are on disk, so part 2
    is blocked until this file has run.
  * THE CALIBRATION of the field. Phase 5b part 1 found the ranking sound and
    the level wrong: at a cut of 0.5 the `exclude` head lights up on 62 percent
    of entirely normal chests. Phase 3 met the same shape on the classifier and
    the answer there was two parameters fitted on the selection split, not a
    new training run. The same answer needs the same data.

THE ONE MISTAKE THIS FILE HAS TO AVOID
--------------------------------------
Fitting anything on images the model was trained on. That would make every
threshold and every calibration curve optimistic, and nothing downstream would
show it.

The defence is not to recompute the inner split. `inner_split` is
deterministic, so recomputing it would probably give the same ids, and
"probably" is the wrong word here. The ids are already on disk: every training
run wrote `sel_f{fold}_s{seed}.csv` with exactly the images it selected on.
This file READS that list. A split that is read cannot drift from the split
that was used.

On top of that the run proves its own provenance. `sel_f*.csv` also carries
`p_sel`, the classifier probability that the SELECTED checkpoint produced on
those images during training. Running the same checkpoint over the same images
has to reproduce it. If it does not, then the weights, the image list, the
transform or the normalisation is not the one the number came from, and the
head field written here would belong to a different model than it claims.

HOW TO READ THE OUTPUT
----------------------
`ok` on all four checks means the fields belong to the run they are named
after. `max |dp|` is the largest per-image difference against the recorded
probability; it is not expected to be exactly zero because the training ran on
DirectML and a rerun may use a different backend, but it has to be small enough
that no image changes side of any plausible threshold.

CLI, from the repository root:
  python rsna\\befunde\\rsna_kopf_sel.py --dml-index 1
  python rsna\\befunde\\rsna_kopf_sel.py --dml-index 1 --arme ex em
  python rsna\\befunde\\rsna_kopf_sel.py --device cpu --folds 0
"""

from __future__ import annotations

import _repo_path  # noqa: F401

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna_train import (HEAD_GRID, RsnaDataset, TwoHeadNet, build_transforms,
                        pick_device, predict)

TAGS = {"ref": "_p5ref", "ex": "_p5head_ex", "em": "_p5head_em"}
DIRS = {"ref": "predictions_p5_ref", "ex": "predictions_final_model",
        "em": "predictions_p5_head_em"}
# Largest per image difference in the probability that still counts as "the
# same model". The recorded values came off DirectML in float32; a rerun on
# another backend differs in the last bits. 1e-3 is three orders of magnitude
# below the smallest threshold spacing anything in this project uses, so a run
# that passes cannot move an image across a decision boundary.
DP_MAX = 1e-3

FINDINGS: list[str] = []


def check(name: str, cond: bool, info: str = "") -> bool:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FINDINGS.append(name)
    return cond


def ergebniszeile(results: Path, arm: str, fold: int) -> pd.Series:
    """The row this fold's weights come from, the LAST one if it ran twice.

    Reading the checkpoint path out of the result row rather than building it
    from a naming scheme: in this project a file name has already claimed to be
    something it was not.
    """
    r = pd.read_csv(results)
    r = r[(r["tag"] == TAGS[arm]) & (r["fold"] == fold)]
    if r.empty:
        raise SystemExit(f"ABORT: no row in {results} for {arm} fold {fold}.")
    return r.iloc[-1]


def ein_fold(arm: str, fold: int, args) -> None:
    pred_dir = Path(DIRS[arm])
    out = pred_dir / f"head_sel_f{fold}_s{args.seed}.npz"
    if out.exists() and not args.force:
        print(f"  {arm} fold {fold}: {out} exists, skipped")
        return

    row = ergebniszeile(args.results, arm, fold)
    if int(row.get("head", 0)) != 1:
        raise SystemExit(f"ABORT: {arm} fold {fold} was trained without a head, "
                         f"there is no field to compute.")
    grid = int(row.get("head_grid", HEAD_GRID))
    ckpt = Path(str(row["ckpt"]))
    if not ckpt.exists():
        raise SystemExit(f"ABORT: {ckpt} missing, named in {args.results}.")

    sel_csv = pred_dir / f"sel_f{fold}_s{args.seed}.csv"
    if not sel_csv.exists():
        raise SystemExit(f"ABORT: {sel_csv} missing. It carries the selection "
                         f"ids and is the only trustworthy source for them.")
    sel = pd.read_csv(sel_csv)
    ids = [str(x) for x in sel["patientId"]]
    labels = {str(p): float(y) for p, y in zip(sel["patientId"], sel["y"])}

    print(f"\n  {arm} fold {fold}: {len(ids)} selection images, grid {grid}, "
          f"weights {ckpt.name}")

    # The selection split must not touch the reporting split. Cheap, and it is
    # the failure that would invalidate every later threshold.
    # ABORT and not a finding. A finding lands in a log, and a log gets read
    # once. An overlap here means every threshold and every calibration built
    # on the result would be fitted on reporting data, and nothing downstream
    # would notice. So the run stops before it computes anything.
    sp = json.loads(Path(args.splits).read_text())
    val = set(sp["folds"][fold]["val"])
    ueberschnitt = sorted(val.intersection(ids))
    if ueberschnitt:
        raise SystemExit(
            f"ABORT: {len(ueberschnitt)} of the {len(ids)} selection images of "
            f"{arm} fold {fold} also sit in the validation split, for example "
            f"{ueberschnitt[:3]}.\nA threshold fitted on them would be fitted "
            f"on the reporting set. Nothing has been computed.")
    print(f"    selection and validation do not overlap ({len(val)} val ids)")

    device, pin, label = pick_device(args.device, args.dml_index)
    print(f"    hardware: {label}")

    # pretrained=False: the ImageNet weights are overwritten by the checkpoint
    # in the next line, so downloading them would cost 45 MB to change nothing.
    net = TwoHeadNet(grid, pretrained=False)
    # weights_only=True: the file is a plain state dict of tensors written by
    # this project, so nothing else has to be unpickled. It also silences the
    # warning that torch prints five times over five folds, and it is the
    # default torch is moving to anyway.
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if "loc.weight" not in state:
        raise SystemExit(f"ABORT: {ckpt} carries no head weights. Either the "
                         f"file belongs to a single headed run or it was "
                         f"overwritten.")
    net.load_state_dict(state)
    net = net.to(device)

    loader = DataLoader(
        RsnaDataset(args.images, ids, labels, build_transforms(args.size, False)),
        batch_size=args.batch, num_workers=args.workers, pin_memory=pin)
    t0 = time.time()
    p, y, fields = predict(net, loader, device, fields=True)
    dt = time.time() - t0

    # ---- provenance, and this is the point of the whole file ---------------
    p_ref = sel["p_sel"].to_numpy()
    dp = float(np.abs(p - p_ref).max())
    korr = float(np.corrcoef(p, p_ref)[0, 1])
    check("the recorded selection probabilities are reproduced", dp < DP_MAX,
          f"max |dp| {dp:.3e}, correlation {korr:.8f}")
    if dp >= DP_MAX:
        print("        These weights do not produce the numbers that were "
              "written")
        print("        beside them. Nothing is saved. Suspects, in order: the "
              "checkpoint")
        print("        was overwritten by a later run, the image folder is not "
              "the one")
        print("        the training used, or the evaluation transform changed.")
        return
    ok_y = check("labels match the recorded ones",
                 bool(np.array_equal(y, sel["y"].to_numpy())))
    ok_f = check("the field has the announced shape",
                 tuple(fields.shape) == (len(ids), grid, grid),
                 str(tuple(fields.shape)))
    if not (ok_y and ok_f):
        print("        Not saved. A file that is written despite a failed check")
        print("        is a file that gets used despite a failed check.")
        return

    np.savez_compressed(out, patientId=np.array(ids),
                        field=fields.astype(np.float32), grid=np.int32(grid))
    print(f"    {dt:.0f} s, {out} written "
          f"({out.stat().st_size / 1e6:.1f} MB)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arme", nargs="+", default=["ex"],
                   choices=sorted(TAGS), help="ex is the winner of phase 5 and "
                                              "the only one anything is built on")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--images", type=Path, default=Path("data/rsna/png512"))
    p.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    p.add_argument("--results", type=Path, default=Path("results_rsna.csv"))
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=0,
                   help="leave at 0 on Windows: spawn reimports torch in every "
                        "worker")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "directml", "cpu"])
    p.add_argument("--dml-index", type=int, default=0,
                   help="1 is the RX 5500 XT. Nothing here depends on the chip, "
                        "the run computes and does not train, but the "
                        "provenance check is tighter on the adapter the "
                        "training used")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    print("=" * 70)
    print("HEAD FIELDS ON THE SELECTION SPLIT")
    print("=" * 70)
    for arm in args.arme:
        for fold in args.folds:
            ein_fold(arm, fold, args)

    print("\n" + "=" * 70)
    if FINDINGS:
        print(f"{len(FINDINGS)} FINDING(S):")
        for f in FINDINGS:
            print(f"  - {f}")
        return 1
    print("No finding. The selection fields belong to the runs they are named")
    print("after, and the threshold for phase 5b part 2 can be set on them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
