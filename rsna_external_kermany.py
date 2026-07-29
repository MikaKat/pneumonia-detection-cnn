"""
Step 10: external validation. The RSNA model on Kermany, inference only.

What this measures
------------------
Every number up to here comes from one dataset. Cross-validation protects
against overfitting to a sample, not against overfitting to an acquisition
setting. The clinical question is the other one: does this work at another
hospital, on another machine, on other patients?

The change of setting is deliberately harsh.

  * RSNA: adults, US emergency departments, DICOM 1024x1024, many bedside
    images, positive rate 0.225.
  * Kermany: children aged 1-5, Guangzhou, JPEG in varying sizes,
    positive rate 0.74.

Age, country, machine, file format and prevalence all change at once. A model
that survives that has learned something about lungs. A drop is a result rather
than a failure, provided the cause is named.

Nothing is trained here. The five existing RSNA checkpoints, that is, the
stored weights of the five cross-validation models, are run as they are.

Four controls, without which the number would be worthless
----------------------------------------------------------
1. The Kermany metadata leak. The JPEG image dimensions alone separate the
   classes at AUC ~0.91. AUC is the probability that a random positive case is
   scored above a random negative one, and 0.5 is chance. Rescaling to 224x224
   turns the dimension difference into a systematic difference in downsampling
   factor and in aspect ratio, a channel that even a foreign model can tap by
   accident. The header score is therefore computed as well, AND the model AUC
   is reported WITHIN its quintiles. That is the stratification RSNA needed
   for ViewPosition.

2. Stretching versus padding to a square. `build_transforms` rescales bluntly
   to 224x224. RSNA images are square, Kermany images are not, so the model
   would be shown distorted lungs: a distortion it never learned, and one that
   correlates with the class (see 1). Both variants are run. The difference
   between them is itself a measurement.

3. Grouped bootstrap. A Kermany patient contributes several images
   (`person17_bacteria_43.jpeg`). Resampling images would pretend these were
   independent cases and would give too narrow an interval. Patient groups are
   drawn instead.

4. Threshold transfer. The threshold found on RSNA is applied unchanged. With a
   prevalence that jumps from 0.225 to 0.74 the result is predictably poor, and
   that is what happens when a model goes to another hospital without
   recalibration. The number belongs in the report, not hidden away.

Interpreting the output
-----------------------
  * External AUC with grouped bootstrap interval. Read it against the internal
    RSNA reference and against the metadata leak printed above it. A value near
    the leak means the ranking may come from image geometry rather than anatomy.
  * AUC stratified by the leak score. This is the leak-adjusted reading, and it
    is averaged with weights given by the DISCORDANT PAIRS of each stratum
    (n_pos x n_neg) rather than by stratum size: a quintile holding a thousand
    positives and a single negative is large but carries almost no information,
    and weighting by size would let its noise dominate the mean. Staying clearly
    above 0.5 means the model sees more than the leak; falling to 0.5 means the
    overall number was the leak.
  * Operating point at the transferred threshold. Discrimination, the ordering
    of cases by risk, and calibration, whether a score of 0.6 really means a
    60 percent chance, travel separately. The ranking can survive the domain
    shift, meaning the change of hospital, machine and patient population, while
    the threshold does not. Sensitivity, specificity, PPV and NPV at an
    unchanged threshold under a prevalence moving from 0.225 to 0.74 report what
    a deployment without recalibration costs, not what the model can rank.

The difference 'pad' minus 'stretch' is read on its own. It separates a
preprocessing fault from a genuine domain shift.

CLI:
  python rsna_external_kermany.py --device directml
  python rsna_external_kermany.py --split test        # only the official test folder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

EXTS = ("*.jpeg", "*.jpg", "*.png")
CLASSES = {"NORMAL": 0, "PNEUMONIA": 1}


# --------------------------------------------------------------------------
# Collecting the data  (torch-free)
# --------------------------------------------------------------------------

def collect_kermany(root: Path, splits: list[str] | None = None) -> pd.DataFrame:
    """All Kermany images with label, original split and patient group.

    The grouping logic comes from `splits.parse_record`, reused rather than
    reimplemented, so that the patient groups are exactly the same as everywhere
    else in the project. Groups defined twice would drift apart, and a grouped
    bootstrap over the wrong groups is no protection at all.
    """
    from splits import parse_record

    rows = []
    for f in sorted(x for x in Path(root).rglob("*")
                    if x.suffix.lower() in {".jpeg", ".jpg", ".png"}):
        rec = parse_record(f.relative_to(root))
        if rec is None:
            continue
        if splits and rec["split"] not in splits:
            continue
        rec["path"] = str(f)
        rows.append(rec)
    return pd.DataFrame(rows)


def read_dims(paths: list[str]) -> pd.DataFrame:
    """Image dimensions only. PIL does not decode for this, so it takes seconds.

    Width, height, aspect ratio and pixel count are the raw material of the
    Kermany metadata leak. They are available before a single model has run and
    are exactly the features the leak classifier is fitted on.
    """
    w, h = [], []
    for p in paths:
        with Image.open(p) as im:
            w.append(im.size[0])
            h.append(im.size[1])
    w, h = np.array(w, float), np.array(h, float)
    return pd.DataFrame({"width": w, "height": h, "aspect": w / h,
                         "pixels": w * h})


def grouped_bootstrap_auc(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                          B: int = 500, seed: int = 0) -> tuple[float, float, float]:
    """AUC with confidence interval from resampled PATIENT GROUPS.

    Resampling images assumes an independence that does not exist: several
    images of the same child are almost the same image. The interval would then
    be too narrow and the external figure would look more certain than it is.

    Returns the point estimate and the 2.5 / 97.5 percentiles. An interval whose
    lower end reaches down to the metadata leak means the external result is not
    distinguishable from the leak.
    """
    from sklearn.metrics import roc_auc_score

    point = float(roc_auc_score(y, p))
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        if len(set(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    if not out:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def header_leak(dims: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                seed: int = 0) -> tuple[float, np.ndarray]:
    """How well do the bare image dimensions separate? Grouped cross-validated.

    Returns the AUC and the out-of-fold score, that is, the score each image
    receives from a fit that never saw it. The score is needed in order to
    report the model AUC WITHIN its quintiles, the stratification that
    establishes whether the model can do more than the leak.

    The null value is 0.5: at 0.5 the file geometry betrays nothing. The higher
    the value, the larger the share of a raw external AUC that could come from
    geometry alone.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    X = dims[["width", "height", "aspect", "pixels"]].values
    oof = np.zeros(len(y))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups):
        m = GradientBoostingClassifier(random_state=seed).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof)), oof


def stratified_by_score(y: np.ndarray, p: np.ndarray, strat: np.ndarray,
                        q: int = 5) -> tuple[float, list]:
    """Model AUC within the quintiles of a nuisance score.

    If it stays clearly above 0.5, the model sees something the leak does not
    already supply. If it drops to 0.5, the overall figure was the leak.

    WEIGHTED BY DISCORDANT PAIRS (n_pos * n_neg), NOT by n.
    That is not a nicety but the difference between a dependable and a
    misleading number. When the nuisance score is strong, and on Kermany it
    separates at AUC 0.916, the upper quintiles are almost pure positive.
    One quintile measured here held 1182 positive and a single negative image.
    Its AUC rests on 1182 pairs that all contain that same one image, which is
    effectively noise. Weighted by n, that noise enters the mean at full weight
    and yields 0.853. Weighted by discordant pairs, that is, by the information
    actually present, the result is 0.885.
    """
    from sklearn.metrics import roc_auc_score

    edges = np.quantile(strat, np.linspace(0, 1, q + 1))
    edges[-1] += 1e-9
    per, num, den = [], 0.0, 0
    for i in range(q):
        sel = (strat >= edges[i]) & (strat < edges[i + 1])
        npos = int(y[sel].sum())
        nneg = int(sel.sum()) - npos
        pairs = npos * nneg
        if pairs == 0 or sel.sum() < 30:
            per.append((i + 1, int(sel.sum()), npos, nneg, pairs,
                        float("nan")))
            continue
        a = float(roc_auc_score(y[sel], p[sel]))
        per.append((i + 1, int(sel.sum()), npos, nneg, pairs, a))
        num += a * pairs
        den += pairs
    return (num / den if den else float("nan")), per


def operating_point(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    tp = int(((p >= thr) & (y == 1)).sum())
    fp = int(((p >= thr) & (y == 0)).sum())
    fn = int(((p < thr) & (y == 1)).sum())
    tn = int(((p < thr) & (y == 0)).sum())
    return {"sens": tp / max(tp + fn, 1), "spec": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1),
            "pos_rate": float((p >= thr).mean())}


# --------------------------------------------------------------------------
# Torch part
# --------------------------------------------------------------------------

class PadToSquare:
    """Pad to a square instead of stretching, filling with the image median.

    Black bars would themselves be a conspicuous feature and would introduce a
    new edge. `MaskCorners` takes the median rather than zero for the same
    reason.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        s = max(w, h)
        med = int(np.median(np.asarray(img.convert("L"))))
        out = Image.new(img.mode, (s, s), med)
        out.paste(img, ((s - w) // 2, (s - h) // 2))
        return out


def build_variants(size: int):
    """The two preprocessings between which a decision has to be made.

    'stretch' reproduces training-time behaviour exactly, 'pad' preserves the
    geometry. The gap between their external AUCs is what tells a preprocessing
    fault apart from a genuine domain shift.
    """
    import torchvision.transforms as T

    from rsna_train import IMNET_MEAN, IMNET_STD

    base = [T.Grayscale(num_output_channels=3), T.ToTensor(),
            T.Normalize(IMNET_MEAN, IMNET_STD)]
    return {
        # exactly what rsna_train.build_transforms(size, False) does
        "stretch": T.Compose([T.Resize((size, size))] + base),
        # geometry-preserving: a Kermany image then looks more like what the
        # RSNA model knows, because RSNA images are square
        "pad": T.Compose([PadToSquare(), T.Resize((size, size))] + base),
    }


class ListDataset:
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("L"))


def predict_all(paths, tf, models, device, batch: int, workers: int):
    """One pass over the data, all five models per batch.

    Five separate passes would decode every JPEG five times, and that costs
    more here than the forward pass of a ResNet18.

    Returns an [n, n_folds] array of per-fold probabilities.
    """
    import torch
    from torch.utils.data import DataLoader

    dl = DataLoader(ListDataset(list(paths), tf), batch_size=batch,
                    num_workers=workers)
    out = [[] for _ in models]
    with torch.no_grad():
        for bi, x in enumerate(dl, 1):
            x = x.to(device)
            for k, m in enumerate(models):
                out[k].append(torch.sigmoid(m(x).squeeze(1)).float().cpu().numpy())
            if bi % 20 == 0:
                print(f"      Batch {bi}/{len(dl)}")
    return np.stack([np.concatenate(o) for o in out], axis=1)     # [n, n_folds]


def load_state_cpu(path: Path):
    """ALWAYS load the checkpoint onto the CPU, never straight onto the device.

    `torch.load(..., map_location=<DirectML device>)` dies: Torch hands the
    torch.device object on to `torch_directml.device()`, and that function
    expects an integer index there. The error then reads

        TypeError: '>=' not supported between instances of 'torch.device' and 'int'

    and looks as though the checkpoint were at fault. The correct route is this
    one anyway: state_dict onto the CPU, model onto the device, then
    `load_state_dict` copies the weights across.

    `weights_only=True` at the same time suppresses the FutureWarning. The
    checkpoint is a plain state_dict, nothing more is needed. The fallback only
    catches old Torch versions without that argument.
    """
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def load_models(folds, seed, device):
    from rsna_train import make_model

    models = []
    for f in folds:
        ck = Path(f"checkpoints/rsna_f{f}_s{seed}.pth")
        if not ck.exists():
            print(f"  Checkpoint missing, skipped: {ck}")
            continue
        m = make_model(device)                      # puts the model on the device
        m.load_state_dict(load_state_cpu(ck))       # copies CPU -> device
        m.eval()
        models.append(m)
    return models


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, default=Path("data/chest_xray"))
    p.add_argument("--split", nargs="*", default=None,
                   help="only these original splits (train/val/test); empty = all")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--thr", type=float, default=None,
                   help="RSNA threshold; default = mean from results_rsna.csv")
    p.add_argument("--out", type=Path,
                   default=Path("predictions_rsna/external_kermany.csv"))
    args = p.parse_args(argv)

    from rsna_train import pick_device

    d = collect_kermany(args.images, args.split)
    if d.empty:
        print(f"ERROR: no images under {args.images}")
        return 2
    y = d.label.values
    groups = d.group.values
    print(f"Kermany: {len(d)} images, {d.group.nunique()} patient groups | "
          f"positive rate {y.mean():.3f}")
    print(f"  per original split: {dict(d.split.value_counts())}")
    print("  (the RSNA positive rate was 0.225; prevalence rises by a factor of 3)")

    # ---- control 1: the metadata leak, BEFORE any model figure -------------
    dims = read_dims(d.path.tolist())
    leak_auc, leak_score = header_leak(dims, y, groups, args.seed)
    print(f"\nMetadata leak (dimensions only, grouped CV): AUC {leak_auc:.3f}")
    print(f"  aspect ratio alone: "
          f"{max(np.corrcoef(dims.aspect, y)[0, 1], -1):+.3f} correlation with class")
    print("  Every model figure below has to be read against this reference.")

    device, _ = pick_device(args.device)
    print(f"\nDevice: {device}")
    models = load_models(args.folds, args.seed, device)
    if not models:
        print("ERROR: no checkpoint found.")
        return 2
    print(f"  {len(models)} checkpoints loaded")

    thr = args.thr
    if thr is None:
        try:
            thr = float(pd.read_csv("results_rsna.csv")["thr"].mean())
        except Exception:
            thr = 0.5
    print(f"  transferred RSNA threshold: {thr:.4f}")

    results = {}
    for name, tf in build_variants(args.size).items():
        print(f"\n  Variant '{name}' ...")
        P = predict_all(d.path.tolist(), tf, models, device, args.batch,
                        args.workers)
        results[name] = P
        for k in range(P.shape[1]):
            d[f"p_{name}_f{args.folds[k]}"] = P[:, k]
        d[f"p_{name}_ens"] = P.mean(axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    d.assign(**{c: dims[c] for c in dims.columns}).to_csv(args.out, index=False)

    # ---- report -----------------------------------------------------------
    print("\n" + "=" * 76)
    print("EXTERNAL AUC: RSNA model, Kermany data, inference only")
    print("=" * 76)
    print(f"  internal reference (RSNA, stratified): 0.845 +- 0.015")
    print(f"  metadata-leak reference here:          {leak_auc:.3f}")
    print()
    from sklearn.metrics import roc_auc_score

    ens_auc = {}
    for name, P in results.items():
        per = [float(roc_auc_score(y, P[:, k])) for k in range(P.shape[1])]
        ens = P.mean(axis=1)
        a, lo, hi = grouped_bootstrap_auc(y, ens, groups, seed=args.seed)
        ens_auc[name] = a
        print(f"  {name:<8} per fold {np.mean(per):.3f} +- {np.std(per, ddof=1):.3f}"
              f"   ensemble {a:.3f} [{lo:.3f} - {hi:.3f}]  (grouped bootstrap)")
        print(f"           single folds: {', '.join(f'{v:.3f}' for v in per)}")

    if len(ens_auc) > 1:
        diff = ens_auc["pad"] - ens_auc["stretch"]
        print(f"\n  'pad' minus 'stretch': {diff:+.3f}")
        print("  A positive value means the model suffers from the distortion")
        print("  that build_transforms produces for non-square images:")
        print("  a preprocessing fault, not a domain problem.")

    best = max(ens_auc, key=ens_auc.get)
    ens = results[best].mean(axis=1)

    print("\n" + "-" * 76)
    print(f"STRATIFIED by the metadata leak (variant '{best}')")
    print("-" * 76)
    s_auc, per = stratified_by_score(y, ens, leak_score)
    print(f"  {'Q':<3}{'n':>7}{'n_pos':>8}{'n_neg':>8}{'disk.Paare':>13}{'AUC':>9}")
    thin = []
    for q, n, npos, nneg, pairs, a in per:
        txt = f"{a:.3f}" if not np.isnan(a) else "  n/a"
        print(f"  {q:<3}{n:>7}{npos:>8}{nneg:>8}{pairs:>13}{txt:>9}")
        if min(npos, nneg) < 30:
            thin.append(q)
    print(f"\n  weighted mean (by discordant pairs): {s_auc:.3f}")
    print(f"  raw, without stratification:        "
          f"{float(roc_auc_score(y, ens)):.3f}")
    print(f"  metadata leak alone:                {leak_auc:.3f}")
    if thin:
        print(f"\n  Note: quintile(s) {thin} are almost pure (fewer than 30 images")
        print("  of the minority class). Their AUC carries hardly any information,")
        print("  which is why the weighting uses discordant pairs and not n.")
        good = [q for q, n, npos, nneg, pr, a in per if min(npos, nneg) >= 30
                and not np.isnan(a)]
        sel = np.zeros(len(y), bool)
        edges = np.quantile(leak_score, np.linspace(0, 1, 6))
        edges[-1] += 1e-9
        for q in good:
            sel |= (leak_score >= edges[q - 1]) & (leak_score < edges[q])
        if sel.sum() > 50 and len(set(y[sel])) > 1:
            print(f"  Only the well-populated quintiles {good}: n={int(sel.sum())}, "
                  f"AUC {float(roc_auc_score(y[sel], ens[sel])):.3f}")
    print("\n  If this stays clearly above 0.5, the model sees more than the leak.")

    print("\n" + "-" * 76)
    print(f"THRESHOLD TRANSFER without recalibration (threshold {thr:.3f})")
    print("-" * 76)
    op = operating_point(y, ens, thr)
    print(f"  sensitivity {op['sens']:.3f} | specificity {op['spec']:.3f} | "
          f"PPV {op['ppv']:.3f} | NPV {op['npv']:.3f}")
    print(f"  fraction predicted positive {op['pos_rate']:.3f} "
          f"against actual {y.mean():.3f}")
    print("  Prevalence jumps from 0.225 to "
          f"{y.mean():.3f}. A transferred threshold cannot fit here.")
    print("  The figure shows what deployment without recalibration costs.")

    print("\n" + "=" * 76)
    print(f"Raw data: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
