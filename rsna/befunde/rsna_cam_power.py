"""
Step 9i: does the heat map point at the infiltrate? Measured on EVERY positive
validation image instead of a sample of 300, from the existing checkpoints.

WHAT THIS WRITES
----------------
For every fold and every variant one CSV with one row per image (`hit`, `mass`,
`area`, `degenerate`), plus `summary.csv` with the per fold rates and
`paired.csv` with the fold by fold difference between the two variants.

WHY IT EXISTS
-------------
The localisation endpoint is the one the project started from: not "is the
classification right" but "does the evidence sit on the pathology". It is
measured against the RSNA bounding boxes, so it is a number and not an opinion.

That number is currently undecided, and the reason is sample size, not the
model. During training Grad-CAM runs on 300 positive images per fold because it
runs on the CPU and would otherwise stretch every run. Over five folds the
difference between baseline and reweighted model came out at -0.135 with a
spread of 0.118, so t = -2.56 against a limit of 2.78. That is "cannot tell",
and it is measured on 300 of about 1030 available positives per fold.

Tripling the sample costs no training at all. The weights are on disk, Grad-CAM
is a backward pass on the CPU, and the images are the same ones the model was
reported on. Nothing about the models changes, only the precision with which
they are looked at.

HOW TO READ IT
--------------
`hit` is the pointing game: does the maximum of the heat map fall inside an
annotated box. Its chance value is `area`, the share of the image the boxes
cover, around 0.11 to 0.12. A hit rate of 0.34 against 0.11 is a factor of
three over chance; the raw number alone means nothing.

`mass` is the share of heat inside the boxes, again against `area`.

`degenerate` marks maps that are zero everywhere. They count as a miss, because
a model that produces an empty map has failed to localise, and dropping them
would flatter exactly the model that produces them.

The verdict is the PAIRED difference per fold, tested over the five folds with
|t| > 2.78 as the five percent limit. Paired means the same fold is compared
with itself, which is the only comparison that is not swamped by the spread
between folds. As a second, more powerful reading, McNemar counts on how many
individual images one variant hits while the other misses; its five percent
limit is chi square > 3.841. The fold level t stays the primary number because
images inside a fold are not independent of each other.

CLI, from the repository root:
  python rsna\\befunde\\rsna_cam_power.py
  python rsna\\befunde\\rsna_cam_power.py --folds 0 --n 200      (quick test)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _repo_path  # noqa: F401  (setzt sys.path fuer die Nachbarordner)

from rsna_train import cam_vs_boxes, load_boxes, make_model

EVERY = 10 ** 9          # cam_vs_boxes takes min(n, available), so this is "all"
T_LIMIT = 2.78           # five percent, four degrees of freedom
CHI2_LIMIT = 3.841       # five percent, one degree of freedom


def load_model(ckpt: Path):
    """Weights to the CPU, always.

    `torch.load(..., map_location=<DirectML device>)` dies with a TypeError
    that looks like a broken checkpoint and is none. Grad-CAM runs on the CPU
    anyway: it needs a backward pass through hooks, which is neither fast nor
    reliable under DirectML.
    """
    model = make_model(torch.device("cpu"))
    try:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def paired_t(values) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    sd = float(v.std(ddof=1))
    t = float(v.mean() / (sd / np.sqrt(len(v)))) if sd > 0 else float("nan")
    return float(v.mean()), sd, t


def mcnemar(a_hit, b_hit) -> tuple[int, int, float]:
    """Counts the images where exactly one of the two variants hits.

    Images both hit or both miss carry no information about a DIFFERENCE and
    drop out, which is the whole point of the test.
    """
    a = np.asarray(a_hit, dtype=bool)
    b = np.asarray(b_hit, dtype=bool)
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    n = only_a + only_b
    chi2 = ((abs(only_a - only_b) - 1) ** 2) / n if n else float("nan")
    return only_a, only_b, float(chi2)


def fingerprint(model) -> float:
    """One number per weight set, so two variants cannot silently be one.

    The first run of this script reported bit identical Grad-CAM numbers for
    the baseline and the reweighted model, which is impossible for two
    different weight sets and was in fact one weight set read twice. See
    `check_provenance` for the cause.
    """
    with torch.no_grad():
        return float(sum(t.double().abs().sum() for t in model.state_dict().values()))


def check_provenance(model, root: Path, ids: list[str], labels: dict,
                     size: int, pred_csv: Path) -> tuple[float, float]:
    """Does this checkpoint still produce the predictions that were reported?

    A checkpoint file says nothing about which run wrote it. Four of the five
    baseline checkpoints turned out to hold the crop models and the fifth the
    reweighted model, because the runs before the `--tag` switch all wrote to
    `rsna_f{fold}_s{seed}.pth`. Nothing in the file names showed it.

    The prediction CSVs are the ground truth here: they were written by the run
    whose numbers are reported. Scoring a handful of those images again and
    correlating with the stored `p_clean` settles in seconds whether the
    weights belong to the numbers.

    Returns (correlation, largest absolute difference). A checkpoint that
    belongs to the run reproduces its predictions exactly, so the correlation
    is 1.0 and the difference is around 1e-7.
    """
    from torch.utils.data import DataLoader
    from rsna_train import RsnaDataset, build_transforms, predict

    stored = pd.read_csv(pred_csv).set_index("patientId")["p_clean"]
    common = [i for i in ids if i in stored.index]
    if not common:
        return float("nan"), float("nan")
    ds = RsnaDataset(root, common, labels, build_transforms(size, False))
    p_now, _ = predict(model, DataLoader(ds, batch_size=32, num_workers=0),
                       torch.device("cpu"))
    p_old = stored.loc[common].to_numpy()
    return float(np.corrcoef(p_now, p_old)[0, 1]), float(np.abs(p_now - p_old).max())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, default=Path("data/rsna/png512"))
    p.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--n", type=int, default=0,
                   help="0 = every positive validation image (the point of this "
                        "script); a small number is for a quick test")
    p.add_argument("--a-tag", default="_base", help="checkpoint suffix of variant A")
    p.add_argument("--a-name", default="Basislinie")
    p.add_argument("--a-dir", type=Path, default=Path("predictions_rsna"),
                   help="the predictions this variant REPORTED; used to "
                        "check that the checkpoint belongs to them")
    p.add_argument("--b-tag", default="_bal10", help="checkpoint suffix of variant B")
    p.add_argument("--b-name", default="balance-view 1.0")
    p.add_argument("--b-dir", type=Path, default=Path("predictions_rsna_bal10"))
    p.add_argument("--min-corr", type=float, default=0.999,
                   help="below this the checkpoint does not belong to the "
                        "reported run and the variant is skipped")
    p.add_argument("--out-dir", type=Path, default=Path("predictions_cam_full"))
    args = p.parse_args()

    sp = json.loads(args.splits.read_text())
    labels = {k: int(v) for k, v in sp["labels"].items()}
    boxes = load_boxes(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_take = args.n if args.n else EVERY

    variants = [(args.a_tag, args.a_name, args.a_dir),
                (args.b_tag, args.b_name, args.b_dir)]
    rows, per_image, prints = [], {}, {}
    t0 = time.time()

    for fold in args.folds:
        val_ids = sp["folds"][fold]["val"]
        for tag, name, pdir in variants:
            ckpt = args.ckpt_dir / f"rsna_f{fold}_s{args.seed}{tag}.pth"
            if not ckpt.exists():
                print(f"  MISSING {ckpt}. Fold {fold} skipped for {name}.")
                continue
            print(f"\nFold {fold}, {name}: {ckpt.name}")
            model = load_model(ckpt)

            fp = fingerprint(model)
            twin = prints.get((fold, round(fp, 3)))
            if twin:
                print(f"  ABORT: identical weights to {twin} in this fold. "
                      f"Two variants cannot be one file.")
                continue
            prints[(fold, round(fp, 3))] = name

            pred_csv = pdir / f"rsna_f{fold}_s{args.seed}.csv"
            if pred_csv.exists():
                probe = [i for i in val_ids if i in boxes][:120]
                corr, dmax = check_provenance(model, args.images, probe, labels,
                                              args.size, pred_csv)
                print(f"  provenance against {pred_csv}: r = {corr:.6f}, "
                      f"largest difference {dmax:.2e}")
                if not (corr >= args.min_corr):
                    print("  ABORT: this checkpoint does not reproduce the "
                          "predictions that were reported for it. It belongs "
                          "to a different run. Fold skipped.")
                    continue
            else:
                print(f"  no {pred_csv}, provenance NOT checked")

            print("  Grad-CAM over the positive validation images (CPU) ...")
            res, d = cam_vs_boxes(model, args.images, val_ids, boxes,
                                  args.size, n_take, args.seed)
            if d.empty:
                print("  no images with boxes, skipped")
                continue
            d = d.sort_values("patientId").reset_index(drop=True)
            d.to_csv(args.out_dir / f"camfull_f{fold}{tag or '_base'}.csv", index=False)
            per_image[(fold, tag)] = d
            rows.append({"fold": fold, "variant": name, "tag": tag, **res})
            print(f"  n {res['cam_n']}   hit {res['cam_hit']:.4f} vs chance "
                  f"{res['cam_area_baseline']:.4f} (lift {res['cam_hit_lift']:+.4f})"
                  f"   mass {res['cam_mass']:.4f}   degenerate "
                  f"{res['cam_degenerate']}   [{(time.time() - t0) / 60:.1f} min]")

    if not rows:
        print("\nnothing measured, no checkpoints found")
        return

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "summary.csv", index=False)

    # ---- paired, fold by fold ------------------------------------------
    paired = []
    for fold in args.folds:
        a = per_image.get((fold, args.a_tag))
        b = per_image.get((fold, args.b_tag))
        if a is None or b is None:
            continue
        m = a.merge(b, on="patientId", suffixes=("_a", "_b"))
        only_a, only_b, chi2 = mcnemar(m["hit_a"], m["hit_b"])
        paired.append({
            "fold": fold, "n_common": len(m),
            "hit_a": m["hit_a"].mean(), "hit_b": m["hit_b"].mean(),
            "d_hit": m["hit_b"].mean() - m["hit_a"].mean(),
            "mass_a": m["mass_a"].mean(), "mass_b": m["mass_b"].mean(),
            "d_mass": m["mass_b"].mean() - m["mass_a"].mean(),
            "deg_a": int(m["degenerate_a"].sum()), "deg_b": int(m["degenerate_b"].sum()),
            "only_a_hits": only_a, "only_b_hits": only_b, "mcnemar_chi2": chi2,
        })
    if not paired:
        print("\nno fold has both variants, no paired comparison")
        return

    dp = pd.DataFrame(paired)
    dp.to_csv(args.out_dir / "paired.csv", index=False)

    print("\n" + "=" * 74)
    print(f"PAIRED: {args.b_name} minus {args.a_name}, every positive image")
    print("=" * 74)
    print(dp.round(4).to_string(index=False))

    if len(dp) < 2:
        print("\n  Only one fold. A spread needs at least two, so there is no "
              "verdict here.\n  This is the shape of a test run; the full run "
              "covers five folds.")
    else:
        for col, label in (("d_hit", "hit rate"), ("d_mass", "mass")):
            mean, sd, t = paired_t(dp[col])
            verdict = "SECURED" if abs(t) > T_LIMIT else "not secured"
            print(f"\n  {label:<10} {mean:+.4f} +- {sd:.4f}   t = {t:6.2f}   "
                  f"{verdict}")
    if dp["mcnemar_chi2"].notna().any():
        worst = dp.loc[dp["mcnemar_chi2"].idxmax()]
        print(f"\n  McNemar per fold, chi square against {CHI2_LIMIT}: "
              f"{', '.join(f'{c:.1f}' for c in dp['mcnemar_chi2'])}")
        print(f"  largest single fold effect in fold {int(worst['fold'])}: "
              f"{int(worst['only_a_hits'])} images only {args.a_name} hits, "
              f"{int(worst['only_b_hits'])} only {args.b_name}")
    else:
        print("\n  McNemar: no image where exactly one variant hits, "
              "nothing to test.")
    print("\n  The fold level t is the verdict. McNemar has more power but "
          "treats\n  images inside a fold as independent, which they are not.")
    print(f"\nsaved: {args.out_dir}/summary.csv, {args.out_dir}/paired.csv, "
          f"per image CSVs")


if __name__ == "__main__":
    main()
