"""
Sanity run on RSNA: one fold, one model, all controls.

A fold is one patient-grouped division of the data into a training part and a
reporting part, and this script runs exactly one of them. It writes a metrics
row to results_rsna.csv, the per-image predictions for the reporting split and
for the inner selection split, a per-epoch history CSV, the Grad-CAM table and
the weights of the selected epoch.

The aim is not the best value but a defensible first one, plus an answer to
four questions:

  1. Does the model beat the header-only baseline of 0.729? AUC is the
     probability that a random pneumonia case is ranked above a random
     non-case, the same quantity as a c-statistic, and 0.729 is what a
     classifier reaches that never sees an image. Anything below means the
     model learned less than the header alone carries.
  2. Does it beat that baseline within one projection? There it sits at 0.553
     (AP) and 0.559 (PA). This is the actual question. The overall AUC still
     contains the AP/PA effect, the stratified one does not.
  3. Does Grad-CAM point at the pathology? Grad-CAM turns the evidence the
     network used into a heatmap over the film. RSNA ships bounding boxes, so
     this is measurable instead of a matter of opinion. It was the starting
     question of the whole project.
  4. Does the model read the burnt-in markers? "PORTABLE" is printed on the AP
     films. The corner ablation, which blanks the four corners and repeats the
     evaluation, answers that.

How to read the output: auc goes against 0.729, auc_stratified against roughly
0.556, and cam_hit against cam_area_baseline, the fraction of image area the
boxes cover. A stratified AUC down at its baseline, or a hit rate no better
than the box area, means the image contributed nothing. A large drop under the
corner ablation means the marker was doing the work.

Carried over from phase 3, where it was learned the expensive way:
  * Checkpoint and threshold both come from an inner, patient-grouped
    selection split. The outer val is only reported, never optimised.
  * `auc_last` and `*_oracle` run along as an optimistic reference so the gap
    stays visible.
  * Every prediction goes to disk as CSV. On Kermany each follow-up question
    otherwise cost a full retraining run.

New against phase 3:
  * No caliper matching. The confounder here is binary and exactly known, so
    the metrics are stratified instead. That costs not a single image, while
    the matching on Kermany discarded two thirds of the data.
  * No `RandomResolution`. That jitter was built against the Kermany zoom
    confounder. Here all images are the same size and it would only add noise.

--balance-view: taking the incentive away instead of the pixels
---------------------------------------------------------------
Three attempts to remove the projection from the images all failed, and the
crop made it worse. Masking rewrites the channel into the lung silhouette,
where 0.692 of 0.714 survives. The crop rewrites it into the magnification
factor: the window side alone predicts AP against PA at 0.685. Per-image
normalisation rewrites it into the relative intensity of the lung, 0.721 rising
to 0.768. A deterministic transform can re-encode information, it cannot delete
it.

The reason the model reads the projection is not that the projection is
visible. It is that the projection is USEFUL: `ViewPosition -> Target` has AUC
0.706, so a model trained on label accuracy has every reason to encode it. The
evidence for the projection is spread over heart size, scapulae, diaphragm
position, framing and sharpness at once, which is why removing any single
carrier changes nothing. The incentive sits in one place only.

`--balance-view` removes the incentive. Every training image is drawn with a
weight that makes projection and label statistically independent in the
training stream, and the weight is the ratio a chi-square test is built on,
expected count under independence divided by observed count:

    w(v, y) = n_v * n_y / (N * n_vy)

On the development set that is AP-negative 1.26, AP-positive 0.59, PA-negative
0.85, PA-positive 2.42. Both MARGINALS stay exactly as they were, the overall
prevalence of 0.225 and the AP to PA ratio; only the association between them
is cut. That is deliberate, because it keeps `pos_weight` valid and the class
imbalance is not corrected twice.

The weights come from the FITTING SPLIT of the current fold alone, never from
the whole development set, or fold information leaks. Selection split and
reporting split are never reweighted, otherwise nothing stays comparable to the
baseline.

How to read the result, written down before the first run:

  * PRIMARY: `AUC(model score -> ViewPosition)` must FALL. Baseline
    0.8166 +- 0.0098 over five folds.
  * SECONDARY: the STRATIFIED AUC must NOT fall. Baseline 0.8449 +- 0.0147.
  * The RAW AUC IS EXPECTED TO FALL, and that is the success, not a
    regression. The 0.880 contains the +0.044 contributed by `ViewPosition`.
    Removing the channel costs roughly that much, so a raw AUC sinking towards
    0.845 is the predicted signature of the intervention working. This has to
    stand here before the run, because afterwards it reads like a step
    backwards.
  * Calibration shifts, so the per-stratum thresholds move. They are searched
    on the selection split, which the reweighting does not touch.

The price is 14 percent of the effective sample size, 19,698 of 22,872 by the
Kish measure. Drawing a PA-positive image 2.42 times does not create 2.42
patients. What it does create, since the draw passes through
`build_transforms` again, is 2.42 differently augmented views of it.

CLI:
  python rsna_train.py --fold 0 --epochs 8 --batch 16 --workers 0
  python rsna_train.py --fold 0 --epochs 8 --batch 16 --workers 0 --balance-view
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, resnet18

IMNET_MEAN, IMNET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BOX_SPACE = 1024          # boxes are given in the original DICOM grid


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

class RsnaDataset(Dataset):
    """IDs instead of paths. The loader builds the path. See rsna_splits.py."""

    def __init__(self, root: Path, ids: list[str], labels: dict[str, int], tf):
        self.root, self.ids, self.labels, self.tf = root, ids, labels, tf

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]
        img = Image.open(self.root / f"{pid}.png").convert("L")
        return self.tf(img), float(self.labels[pid])


class MaskCorners:
    """Sets the four image corners to the median. Ablation against marker reading.

    The AP films carry "PORTABLE", side markers and arrows inside the image.
    That is a direct visual proxy for the acquisition type, and the acquisition
    type is the whole confounder. A crude statistical test (share of bright
    pixels in the corners) finds nothing, but a convolutional network reads
    text better than a brightness threshold, so this gets tried rather than
    argued about.

    The median and not black, on purpose: a black patch is a conspicuous
    feature in itself and would introduce a new edge.
    """

    def __init__(self, frac: float = 0.18):
        self.frac = frac

    def __call__(self, img: Image.Image) -> Image.Image:
        a = np.asarray(img).copy()
        k = int(min(a.shape[:2]) * self.frac)
        med = int(np.median(a))
        a[:k, :k] = med; a[:k, -k:] = med; a[-k:, :k] = med; a[-k:, -k:] = med
        return Image.fromarray(a)


def build_transforms(size: int, train: bool):
    # NO horizontal flipping: it creates situs inversus, mirrors the cardiac
    # silhouette and contradicts the side marker printed in the image.
    base = [T.Grayscale(num_output_channels=3), T.ToTensor(),
            T.Normalize(IMNET_MEAN, IMNET_STD)]
    if not train:
        return T.Compose([T.Resize((size, size))] + base)
    return T.Compose([
        T.Resize((size, size)),
        T.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.93, 1.07)),
        T.ColorJitter(brightness=0.15, contrast=0.15),
    ] + base)


def view_balance_weights(y: np.ndarray, vp: np.ndarray,
                         strength: float = 1.0) -> np.ndarray:
    """Per-image sampling weights that decouple projection from label.

    Returns one weight per image. The weight depends only on which cell of the
    projection-by-label table the image falls into:

        w(v, y) = [ n_v * n_y / (N * n_vy) ] ** strength

    which is the expected count of that cell under independence divided by its
    observed count, the same ratio a chi-square test is built from. Cells that
    are over-represented relative to independence are drawn less often, cells
    that are under-represented more often.

    Two properties make this the mild version rather than the brutal one, and
    both are checked in `test_rsna_train.py`:

      * The MARGINALS are preserved exactly. Weighted, the projections keep
        their sizes and the overall prevalence stays at its original value.
        Only the association between the two is removed. That is why
        `pos_weight` in the loss can stay untouched: the class imbalance is
        still there and is still corrected exactly once.
      * The weights SUM TO N. So `num_samples=len(ids)` keeps epochs the same
        length and the runtime per epoch does not change.

    Balancing all four cells to equal size would also work but is the wrong
    trade here: it costs half the effective sample size (11,772 of 22,872 by
    the Kish measure) against 19,698 for this version, and it would additionally
    require setting `pos_weight` to 1.

    Any value of `vp` forms its own stratum, so "unknown" is handled without a
    special case. Very small strata get extreme weights and are reported by the
    caller rather than silently clipped: clipping would quietly leave part of
    the association in place while the printed intent says it is gone.

    STRENGTH is the dial, and it exists because the full correction is not
    free. On fold 0 it moved the primary endpoint by -0.070 and cost Grad-CAM
    localisation, and the whole loss sat in AP: hit rate 0.596 to 0.342 there,
    against 0.306 to 0.333 in PA. AP positives are drawn 0.59 times and PA
    positives 2.42, so the model trades localisation in the majority
    projection for localisation in the minority one.

    `strength` raises every weight to that power and renormalises, so 0 is the
    untouched baseline, 1 is full independence, and values in between trade the
    two effects off against each other. `residual_view_label_auc` reports what
    a given setting leaves standing, BEFORE an epoch is spent on it. Running
    0.5 next to 0 and 1 turns a single point into a dose-response curve, which
    is the stronger form of evidence: it shows the cost is a dial and not an
    accident.
    """
    y = np.asarray(y).astype(int)
    vp = np.asarray(vp).astype(str)
    n = len(y)
    if n == 0:
        return np.zeros(0, dtype=float)
    w = np.ones(n, dtype=float)
    for v in np.unique(vp):
        for c in np.unique(y):
            cell = (vp == v) & (y == c)
            n_vy = int(cell.sum())
            if n_vy == 0:
                continue
            w[cell] = (vp == v).sum() * (y == c).sum() / (n * n_vy)
    if strength != 1.0:
        w = w ** float(strength)
        # Renormalise, because w ** a no longer sums to n. Without this the
        # epoch would silently change length with the setting and any
        # comparison between settings would confound dose with training time.
        w *= n / w.sum()
    return w


def residual_view_label_auc(y: np.ndarray, vp: np.ndarray,
                            w: np.ndarray | None = None) -> float:
    """AUC(ViewPosition -> label) in the WEIGHTED training stream.

    This is the dose axis of the dose-response curve, and it is exact rather
    than estimated: both variables are binary, so the AUC follows from the
    2x2 table without fitting anything.

        AUC = P(AP | y=1) P(PA | y=0) + 0.5 [ P(AP|y=1) P(AP|y=0)
                                            + P(PA|y=1) P(PA|y=0) ]

    the usual convention that ties count half. Unweighted on the development
    set this returns 0.706, the documented value of the confounder; at
    `strength = 1` it returns 0.500 by construction. What it answers is how
    much of the association a chosen setting still leaves in the training
    stream, and it answers it in milliseconds instead of 2.3 hours.
    """
    y = np.asarray(y).astype(int)
    vp = np.asarray(vp).astype(str)
    w = np.ones(len(y), float) if w is None else np.asarray(w, float)
    ap = vp == "AP"
    m1, m0 = y == 1, y == 0
    if w[m1].sum() == 0 or w[m0].sum() == 0:
        return float("nan")
    p1 = w[m1 & ap].sum() / w[m1].sum()          # P(AP | positive)
    p0 = w[m0 & ap].sum() / w[m0].sum()          # P(AP | negative)
    return float(p1 * (1 - p0) + 0.5 * (p1 * p0 + (1 - p1) * (1 - p0)))


def balance_report(w: np.ndarray, y: np.ndarray, vp: np.ndarray) -> list[dict]:
    """One row per cell, for printing. Separate so it can be tested."""
    y = np.asarray(y).astype(int)
    vp = np.asarray(vp).astype(str)
    out = []
    for v in sorted(set(vp)):
        for c in sorted(set(y)):
            cell = (vp == v) & (y == c)
            if not cell.any():
                continue
            out.append({"viewpos": v, "target": int(c), "n": int(cell.sum()),
                        "weight": float(w[cell][0])})
    return out


def effective_n(w: np.ndarray) -> float:
    """Kish effective sample size. n itself when all weights are equal.

    This is the honest price tag of the reweighting. It answers the question
    that oversampling always invites: no, drawing an image 2.42 times does not
    turn it into 2.42 patients.
    """
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / np.sum(w ** 2)) if w.size else 0.0


def train_loader_kwargs(weights: np.ndarray | None, seed: int) -> dict:
    """The DataLoader arguments that differ between the two modes.

    Takes the FINISHED weight array rather than the ingredients for it. That
    is not a stylistic choice, it is the fix for a bug that cost 74 minutes of
    compute: this function used to recompute the weights from `y`, `vp` and a
    `strength` argument, `main` printed its table from one call and built the
    sampler from another, and the second call was left without the strength
    parameter. The run announced `strength 0.5` on screen and trained at 1.0.
    The training curve came out bit-identical to the previous run, which is the
    only reason it was caught at all.

    With one array there is one source of truth. The table and the sampler
    cannot disagree, because there is nothing left to disagree about.

    `weights=None` means no balancing: this returns `{"shuffle": True}`, no
    torch generator is created and the global RNG is not advanced. The baseline
    of 0.8166 has to stay reproducible from the same file, so the paired
    comparison differs in the tested quantity and in nothing else.

    `WeightedRandomSampler` is imported here and not at module level because
    `test_rsna_train.py` runs with a stubbed torch when no GPU stack is
    installed, and a top-level import of a name the stub does not carry would
    break the whole test file over a function it never calls.
    """
    if weights is None:
        return {"shuffle": True}
    from torch.utils.data import WeightedRandomSampler
    # An explicit generator, so the draw does not depend on how much torch
    # randomness anything else consumed first.
    g = torch.Generator().manual_seed(seed)
    return {"sampler": WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights), replacement=True, generator=g)}


PERTURBATIONS = {
    "clean":       lambda s: [T.Resize((s, s))],
    "corners":     lambda s: [MaskCorners(), T.Resize((s, s))],
    "zoom_in":     lambda s: [T.Resize((int(s * 1.15),) * 2), T.CenterCrop(s)],
    "shift":       lambda s: [T.Resize((s, s)), T.RandomAffine(0, translate=(0.08, 0.08))],
    "rotate":      lambda s: [T.Resize((s, s)), T.RandomRotation(12)],
    "low_contr":   lambda s: [T.Resize((s, s)), T.ColorJitter(contrast=(0.6, 0.6))],
    "bright":      lambda s: [T.Resize((s, s)), T.ColorJitter(brightness=(1.35, 1.35))],
    "blur":        lambda s: [T.Resize((s, s)), T.GaussianBlur(5, sigma=1.6)],
    "lowres":      lambda s: [T.Resize((int(s * 0.45),) * 2), T.Resize((s, s))],
}


def perturbed_transform(size: int, name: str):
    return T.Compose(PERTURBATIONS[name](size) +
                     [T.Grayscale(num_output_channels=3), T.ToTensor(),
                      T.Normalize(IMNET_MEAN, IMNET_STD)])


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def pick_device(name: str):
    """DirectML is the only GPU path on this hardware (RX 5500 XT, RDNA1)."""
    if name in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda"), True
    if name in ("auto", "directml"):
        try:
            import torch_directml
            if torch_directml.is_available():
                return torch_directml.device(), False
        except ImportError:
            if name == "directml":
                raise SystemExit("torch-directml is missing:  pip install torch-directml")
    if name == "directml":
        raise SystemExit("torch-directml finds no device.")
    return torch.device("cpu"), False


def make_model(device):
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m.to(device)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    p, y = [], []
    for x, t in loader:
        logit = model(x.to(device, non_blocking=True)).squeeze(1)
        p.append(torch.sigmoid(logit).float().cpu().numpy())
        y.append(t.numpy())
    return np.concatenate(p), np.concatenate(y)


def bce_from_probs(y: np.ndarray, p: np.ndarray, pos_weight: float = 1.0) -> float:
    """The same loss as in training, but computed from probabilities.

    `predict` returns probabilities, not logits. Instead of changing the
    signature (and with it every caller), the loss is recomputed here, with the
    same `pos_weight` as `BCEWithLogitsLoss`. Otherwise the training curve and
    the selection curve are not comparable and the learning curve shows a gap
    between the two that does not exist.

    Clipping at 1e-7 catches p = 0 and p = 1. Without it a single saturated
    prediction returns inf and the whole curve is empty.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean(-(pos_weight * y * np.log(p) + (1 - y) * np.log(1 - p))))


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tpr = np.cumsum(ys) / max(ys.sum(), 1)
    fpr = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
    return float(ps[int(np.argmax(tpr - fpr))])


def scores(y: np.ndarray, p: np.ndarray, thr: float | None = None) -> dict:
    """Ranking metrics plus sensitivity and specificity at a GIVEN threshold.

    `*_oracle` picks the threshold on the same set that is reported on. It is
    written out on purpose as the optimistic counterpart, so that the gap to
    the honest number stays visible.
    """
    out = {"auc": float(roc_auc_score(y, p)),
           "auprc": float(average_precision_score(y, p))}
    t_or = youden_threshold(y, p)
    out["sens_oracle"] = float(((p >= t_or) & (y == 1)).sum() / max((y == 1).sum(), 1))
    out["spec_oracle"] = float(((p < t_or) & (y == 0)).sum() / max((y == 0).sum(), 1))
    t = t_or if thr is None else thr
    out["thr"] = float(t)
    out["sens"] = float(((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1))
    out["spec"] = float(((p < t) & (y == 0)).sum() / max((y == 0).sum(), 1))
    return out


def stratified_scores(y: np.ndarray, p: np.ndarray, vp: np.ndarray,
                      thr: float, thr_by_view: dict[str, float] | None = None
                      ) -> dict:
    """AUC per projection, plus sens/spec at the global and the per-stratum threshold.

    The overall AUC contains the AP/PA effect (header baseline 0.729). Within
    one projection that effect drops out and the baseline is about 0.556. Only
    this stratified number says anything about radiology.

    Why two thresholds on top of that: in the first run the AUC was practically
    identical (0.818 AP against 0.824 PA), yet at ONE threshold sensitivity was
    0.839 in AP and 0.498 in PA. The same number behaves like two different
    tests in the two projections. In PA films half the pneumonias would have
    been missed. That is not a model error but the prevalence difference (0.383
    against 0.093) turning into sensitivity through a fixed threshold. Both are
    reported so the effect stays visible instead of being averaged away.
    """
    out = {}
    for v in ("AP", "PA"):
        m = vp == v
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        pos, neg = y[m] == 1, y[m] == 0
        out[f"auc_{v}"] = float(roc_auc_score(y[m], p[m]))
        out[f"n_{v}"] = int(m.sum())
        out[f"pos_{v}"] = float(y[m].mean())
        out[f"sens_{v}"] = float(((p[m] >= thr) & pos).sum() / max(pos.sum(), 1))
        out[f"spec_{v}"] = float(((p[m] < thr) & neg).sum() / max(neg.sum(), 1))
        if thr_by_view and v in thr_by_view:
            t = thr_by_view[v]
            out[f"thr_{v}"] = float(t)
            out[f"sens_{v}_strat"] = float(((p[m] >= t) & pos).sum() / max(pos.sum(), 1))
            out[f"spec_{v}_strat"] = float(((p[m] < t) & neg).sum() / max(neg.sum(), 1))

    if "auc_AP" in out and "auc_PA" in out:
        # weighted mean of the strata: the AUC that would remain if AP and PA
        # were equally frequent, the confounder-free value
        w = np.array([out["n_AP"], out["n_PA"]], float)
        out["auc_stratified"] = float(
            (out["auc_AP"] * w[0] + out["auc_PA"] * w[1]) / w.sum())
        # The direct measure of the problem: how far apart do the
        # sensitivities of the two projections sit?
        out["sens_gap"] = float(abs(out["sens_AP"] - out["sens_PA"]))
        if "sens_AP_strat" in out and "sens_PA_strat" in out:
            out["sens_gap_strat"] = float(
                abs(out["sens_AP_strat"] - out["sens_PA_strat"]))
    return out


def inner_split(ids: list[str], labels, vp: dict[str, str], seed: int,
                n_splits: int) -> tuple[list[str], list[str]]:
    """Splits fold["train"] into a fit part and a selection part, stratified as outside.

    Picking the checkpoint by AUC on the OUTER val and then reporting that same
    AUC makes every number a maximum over all epochs on the reporting data. On
    Kermany the ceiling hid that. At an AUC of about 0.85 it shifts the number
    by the order of magnitude of the effects one wants to measure.

    Stratification here is again by label x ViewPosition, otherwise the AP/PA
    ratio of the selection split drifts away from val and the threshold no
    longer fits.
    """
    strat = np.array([f"{labels[i]}|{vp[i]}" for i in ids])
    g = np.array(ids)                      # one image per patient
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fit_i, sel_i = next(iter(sgkf.split(np.zeros(len(ids)), strat, g)))
    assert not (set(g[fit_i]) & set(g[sel_i])), "group leak in the inner split!"
    return [ids[i] for i in fit_i], [ids[i] for i in sel_i]


# --------------------------------------------------------------------------
# Grad-CAM against the bounding boxes
# --------------------------------------------------------------------------

def load_boxes(csv_dir: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    df = pd.read_csv(Path(csv_dir) / "stage_2_train_labels.csv")
    df = df[df["Target"] == 1].dropna(subset=["x", "y", "width", "height"])
    out: dict[str, list] = {}
    for pid, x, y, w, h in df[["patientId", "x", "y", "width", "height"]].values:
        out.setdefault(pid, []).append((float(x), float(y), float(w), float(h)))
    return out


def cam_vs_boxes(model, root: Path, ids: list[str], boxes: dict, size: int,
                 n: int, seed: int) -> tuple[dict, pd.DataFrame]:
    """Measures whether the heatmap points at the infiltrate.

    Two measures, both against a chance baseline:

      hit    Does the maximum of the heatmap fall inside a box? ("pointing game")
             Chance baseline = area fraction of the boxes.
      mass   Which share of the heatmap mass lies inside the boxes?
             Chance baseline is again the area fraction.

    The area fraction MUST be reported with them. The boxes cover a substantial
    part of the image; a hit rate of 0.6 sounds good and would be next to
    nothing at an area fraction of 0.55. Without that baseline the number is
    worthless, which is the mistake Grad-CAM figures in presentations usually
    make.

    Runs on the CPU: Grad-CAM needs a backward pass through hooks, and that is
    neither fast nor reliable under DirectML.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

    rng = np.random.default_rng(seed)
    pos = [i for i in ids if i in boxes]
    if not pos:
        return {}, pd.DataFrame()
    pick = rng.choice(pos, min(n, len(pos)), replace=False)

    m = model.to("cpu").eval()
    cam = GradCAM(model=m, target_layers=[m.layer4[-1]])
    tf = build_transforms(size, False)
    s = size / BOX_SPACE

    rows = []
    for j, pid in enumerate(pick, 1):
        img = Image.open(root / f"{pid}.png").convert("L")
        x = tf(img).unsqueeze(0)
        heat = cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(1)])[0]
        heat = np.clip(heat, 0, None)
        if heat.sum() <= 0:
            continue

        mask = np.zeros_like(heat, bool)
        for bx, by, bw, bh in boxes[pid]:
            x0, y0 = int(bx * s), int(by * s)
            x1, y1 = int((bx + bw) * s), int((by + bh) * s)
            mask[max(y0, 0):y1, max(x0, 0):x1] = True

        area = float(mask.mean())
        yx = np.unravel_index(int(np.argmax(heat)), heat.shape)
        rows.append({"patientId": pid, "hit": bool(mask[yx]),
                     "mass": float(heat[mask].sum() / heat.sum()),
                     "area": area, "n_boxes": len(boxes[pid])})
        if j % 100 == 0:
            print(f"      Grad-CAM {j}/{len(pick)}")

    d = pd.DataFrame(rows)
    if d.empty:
        return {}, d
    res = {
        "cam_n": int(len(d)),
        "cam_hit": float(d["hit"].mean()),
        "cam_mass": float(d["mass"].mean()),
        "cam_area_baseline": float(d["area"].mean()),
        "cam_hit_lift": float(d["hit"].mean() - d["area"].mean()),
        "cam_mass_lift": float(d["mass"].mean() - d["area"].mean()),
    }
    return res, d


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, default=Path("data/rsna/png512"))
    p.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=0,
                   help="leave at 0 on Windows: spawn reimports torch in every worker")
    p.add_argument("--inner-splits", type=int, default=6)
    p.add_argument("--balance-view", action="store_true",
                   help="draw training images so that ViewPosition and label "
                        "are independent; see the module header for the "
                        "pre-registered reading of the result")
    p.add_argument("--balance-strength", type=float, default=1.0,
                   help="dose of the correction: 0 is the baseline, 1 full "
                        "independence, values between trade the confounder "
                        "off against the Grad-CAM loss in AP")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "directml", "cpu"])
    p.add_argument("--cam-n", type=int, default=300, help="0 = skip Grad-CAM")
    p.add_argument("--out", type=Path, default=Path("results_rsna.csv"))
    p.add_argument("--pred-dir", type=Path, default=Path("predictions_rsna"))
    p.add_argument("--tag", default="",
                   help="suffix for the checkpoint file, e.g. _balview. The "
                        "checkpoint name used to depend on fold and seed "
                        "alone, so every variant silently overwrote the "
                        "previous one; the crop runs took the baseline "
                        "checkpoints with them and nobody noticed.")
    p.add_argument("--history", type=Path, default=None,
                   help="per-epoch history (default: "
                        "<pred-dir>/history_f{fold}_s{seed}.csv)")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    sp = json.loads(args.splits.read_text())
    labels = {k: int(v) for k, v in sp["labels"].items()}
    vpmap = sp["viewpos"]
    fold = sp["folds"][args.fold]
    device, pin = pick_device(args.device)

    fit_ids, sel_ids = inner_split(fold["train"], labels, vpmap,
                                   args.seed, args.inner_splits)
    val_ids = fold["val"]
    y_fit = np.array([labels[i] for i in fit_ids])

    print(f"\nFold {args.fold}, Seed {args.seed}, Device {device}")
    print(f"  fit {len(fit_ids)} (pos {y_fit.mean():.3f}) | sel {len(sel_ids)} "
          f"| val {len(val_ids)}")
    print(f"  Targets: overall AUC > 0.729 (header baseline), "
          f"per projection > ~0.556")
    if device.type == "cpu":
        print("  WARNING: CPU. On AMD under Windows:  pip install torch-directml")

    # Reweighting uses the FITTING split of this fold only. Weights taken from
    # the whole development set would carry information out of the other folds.
    vp_fit = np.array([vpmap.get(i, "?") for i in fit_ids])
    # Computed ONCE. Everything below, the printed table and the sampler alike,
    # reads this one array; see train_loader_kwargs for why that matters.
    w_fit = (view_balance_weights(y_fit, vp_fit, args.balance_strength)
             if args.balance_view else None)
    if args.balance_view:
        print(f"\n  --balance-view at strength {args.balance_strength:g}: "
              f"ViewPosition and label decoupled in the training stream")
        print(f"    {'projection':<12}{'label':>7}{'n':>8}{'draws/image':>14}")
        for r in balance_report(w_fit, y_fit, vp_fit):
            flag = "   <-- small stratum" if r["n"] < 50 else ""
            print(f"    {r['viewpos']:<12}{r['target']:>7}{r['n']:>8}"
                  f"{r['weight']:>13.2f}x{flag}")
        print(f"    effective sample size {effective_n(w_fit):.0f} of "
              f"{len(w_fit)} (Kish)")
        # The dose, exact and computed before a single epoch is spent.
        print(f"    AUC(ViewPosition -> label) in the stream: "
              f"{residual_view_label_auc(y_fit, vp_fit, w_fit):.3f}   "
              f"(untouched {residual_view_label_auc(y_fit, vp_fit):.3f}, "
              f"fully decoupled 0.500)")
        print("    PRIMARY endpoint: AUC(score -> ViewPosition) must fall "
              "from 0.8166 +- 0.0098.")
        print("    The RAW AUC is expected to fall by about 0.044 as well. "
              "That is the success signature,")
        print("    not a regression. See the module header.")

    tr = DataLoader(RsnaDataset(args.images, fit_ids, labels,
                               build_transforms(args.size, True)),
                    batch_size=args.batch, num_workers=args.workers,
                    pin_memory=pin, drop_last=True,
                    **train_loader_kwargs(w_fit, args.seed))
    sel = DataLoader(RsnaDataset(args.images, sel_ids, labels,
                                build_transforms(args.size, False)),
                     batch_size=args.batch * 2, num_workers=args.workers)
    va = DataLoader(RsnaDataset(args.images, val_ids, labels,
                               build_transforms(args.size, False)),
                    batch_size=args.batch * 2, num_workers=args.workers)

    model = make_model(device)
    # Positive rate 0.225. The imbalance tips the other way than on Kermany
    # (0.74), so pos_weight is > 1 instead of < 1.
    pos_weight = torch.tensor([(y_fit == 0).sum() / max((y_fit == 1).sum(), 1)],
                              dtype=torch.float32, device=device)
    print(f"  pos_weight {pos_weight.item():.2f}")
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(len(tr), 1))

    # Per-epoch history. A single result row at the end leaves no learning
    # curve to show, and a run that dies in hour four leaves nothing at all.
    #
    # What is logged is the SELECTION split (sel), not the reporting set (val).
    # That is the point rather than a saving in the wrong place: sel is exempt
    # from fitting, so the gap between training loss and sel loss shows the
    # overfitting in full. A curve on val would also be an invitation to pick
    # the epoch afterwards, the circular reasoning this project avoids
    # everywhere else.
    hist_path = (args.history if args.history is not None else
                 args.pred_dir / f"history_f{args.fold}_s{args.seed}.csv")
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []

    def write_history() -> None:
        """Written after EVERY epoch, not at the end. An aborted 24-hour run
        should still leave its curve behind."""
        pd.DataFrame(history).to_csv(hist_path, index=False)

    # The checkpoint used to be written once, after training, Grad-CAM and all
    # the perturbations. A machine that goes to sleep in hour two therefore
    # threw away every trained weight and left only the curve. It is written
    # after every improvement now: 45 MB against 74 minutes is not a trade
    # worth thinking about.
    ckpt = Path(f"checkpoints/rsna_f{args.fold}_s{args.seed}{args.tag}.pth")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if ckpt.exists():
        print(f"  NOTE: {ckpt} exists "
              f"({(time.time() - ckpt.stat().st_mtime) / 3600:.1f} h old) and "
              f"is being replaced. Use --tag to keep variants apart.")

    best_sel, best_state, best_ep = -1.0, None, -1
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        # Read the LR BEFORE the epoch. OneCycleLR advances after every step,
        # so after the loop it would hold the rate of the next epoch and the
        # curve would be shifted by one epoch.
        lr_now = float(sched.get_last_lr()[0])
        # Accumulate on the device and fetch it ONCE per epoch. `float(loss)`
        # inside the loop would wait on the device at every batch, roughly 480
        # synchronisation points per epoch just to record one number for the
        # curve. `drop_last=True` makes all batches the same size, so the mean
        # of the means is the right one.
        loss_sum = torch.zeros((), device=device)
        n_batch = 0
        for x, t in tr:
            x, t = x.to(device, non_blocking=True), t.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x).squeeze(1), t)
            loss.backward()
            opt.step(); sched.step()
            loss_sum += loss.detach(); n_batch += 1
        train_loss = float(loss_sum.cpu()) / max(n_batch, 1)
        ps, ys = predict(model, sel, device)
        a = roc_auc_score(ys, ps)
        improved = a > best_sel
        if improved:
            best_sel, best_ep = a, ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            # Write through a temporary file and rename. A power cut in the
            # middle of torch.save would otherwise leave a truncated
            # checkpoint, which is worse than none: it loads far enough to
            # look plausible.
            tmp = ckpt.with_suffix(".pth.tmp")
            torch.save(best_state, tmp)
            tmp.replace(ckpt)
        dt = time.time() - t0
        history.append({
            "fold": args.fold, "seed": args.seed, "epoch": ep + 1,
            "train_loss": train_loss,
            "sel_loss": bce_from_probs(ys, ps, float(pos_weight.item())),
            "sel_auc": float(a),
            "lr": lr_now, "sec": dt,
            "is_best": int(improved),
        })
        write_history()
        print(f"  epoch {ep + 1}/{args.epochs}  sel AUC {a:.4f}  "
              f"train loss {history[-1]['train_loss']:.4f}  "
              f"sel loss {history[-1]['sel_loss']:.4f}"
              f"{'  <-- best so far' if improved else ''}  "
              f"[{dt:.0f}s, ~{dt * (args.epochs - ep - 1) / 60:.0f} min left]")

    p_last, y = predict(model, va, device)
    auc_last = float(roc_auc_score(y, p_last))

    model.load_state_dict(best_state)
    p_sel, y_sel = predict(model, sel, device)
    thr = youden_threshold(y_sel, p_sel)          # threshold NOT from the reporting set
    # ... and per projection likewise on the selection split, not on val. A
    # per-stratum threshold searched on the reporting set would be the same
    # circular reasoning as the global one and would overstate the gain.
    vp_sel = np.array([vpmap[i] for i in sel_ids])
    thr_by_view = {v: youden_threshold(y_sel[vp_sel == v], p_sel[vp_sel == v])
                   for v in ("AP", "PA")
                   if (vp_sel == v).sum() >= 50
                   and len(np.unique(y_sel[vp_sel == v])) > 1}
    p_val, y = predict(model, va, device)

    vp = np.array([vpmap[i] for i in val_ids])
    res = scores(y, p_val, thr)
    res.update(stratified_scores(y, p_val, vp, thr, thr_by_view))
    res.update({"fold": args.fold, "seed": args.seed, "epochs": args.epochs,
                "auc_last": auc_last, "auc_sel": float(best_sel),
                "best_epoch": best_ep + 1, "n_fit": len(fit_ids),
                "n_sel": len(sel_ids), "n_val": len(val_ids),
                # Goes into results_rsna.csv so that a row can never be
                # mistaken for a baseline row later on.
                "balance_view": int(args.balance_view),
                "balance_strength": (float(args.balance_strength)
                                     if args.balance_view else 0.0),
                # Both derived from the SAME w_fit the sampler used, not
                # recomputed. A recomputation is how the strength argument got
                # lost in the first place.
                "balance_residual_auc": residual_view_label_auc(y_fit, vp_fit,
                                                                w_fit),
                "n_fit_effective": (effective_n(w_fit) if w_fit is not None
                                    else float(len(fit_ids)))})

    # The primary endpoint, computed here as well so it is visible in the run
    # instead of only after rsna_crop_compare.py. Same definition as
    # `rsna_crop_compare.score_to_view`, deliberately not folded to
    # max(a, 1 - a): the direction carries meaning and 0.5 is the floor of the
    # channel, not of the number.
    m_vp = np.isin(vp, ["AP", "PA"])
    res["auc_view"] = (float(roc_auc_score((vp[m_vp] == "AP").astype(int),
                                           p_val[m_vp]))
                       if m_vp.sum() and len(set(vp[m_vp])) > 1 else float("nan"))

    print(f"\n  AUC overall     {res['auc']:.4f}   (last epoch {auc_last:.4f}, "
          f"header baseline 0.729)")
    for v in ("AP", "PA"):
        if f"auc_{v}" in res:
            print(f"  AUC {v} only     {res[f'auc_{v}']:.4f}   "
                  f"(n={res[f'n_{v}']}, pos={res[f'pos_{v}']:.3f}, "
                  f"baseline ~0.556)")
    if "auc_stratified" in res:
        print(f"  AUC stratified  {res['auc_stratified']:.4f}  <-- the honest number")
    if res["auc_view"] == res["auc_view"]:            # not NaN
        d = res["auc_view"] - 0.8166
        print(f"  AUC score->view {res['auc_view']:.4f}   ({d:+.4f} against the "
              f"baseline 0.8166 +- 0.0098)  <-- PRIMARY endpoint, must fall")
    print(f"  Sens {res['sens']:.3f} / Spec {res['spec']:.3f} "
          f"(oracle {res['sens_oracle']:.3f}/{res['spec_oracle']:.3f})")

    # The core question: does ONE threshold behave the same in both projections?
    if "sens_gap" in res:
        print(f"\n  Threshold           {'global':>22}   {'per projection':>22}")
        for v in ("AP", "PA"):
            g = f"Sens {res[f'sens_{v}']:.3f} Spez {res[f'spec_{v}']:.3f}"
            s = (f"Sens {res[f'sens_{v}_strat']:.3f} Spez {res[f'spec_{v}_strat']:.3f}"
                 f" @{res[f'thr_{v}']:.3f}" if f"sens_{v}_strat" in res else "-")
            print(f"    {v:<16}{g:>22}   {s:>22}")
        line = f"    {'Sens-Luecke':<16}{res['sens_gap']:>22.3f}"
        if "sens_gap_strat" in res:
            line += f"   {res['sens_gap_strat']:>22.3f}"
        print(line)
        print("    A fixed threshold at unequal prevalence (0.383 vs 0.093) is")
        print("    effectively a different test in the two projections.")

    # ---- Perturbations, above all the corner ablation ------------------
    preds = {"patientId": list(val_ids), "y": y.tolist(), "viewpos": vp.tolist(),
             "p_clean": p_val.tolist(), "p_last_epoch": p_last.tolist()}
    print()
    for name in [n for n in PERTURBATIONS if n != "clean"]:
        torch.manual_seed(args.seed); random.seed(args.seed)
        ds = RsnaDataset(args.images, val_ids, labels,
                         perturbed_transform(args.size, name))
        pp, yy = predict(model, DataLoader(ds, batch_size=args.batch * 2,
                                           num_workers=args.workers), device)
        res[f"auc_{name}"] = float(roc_auc_score(yy, pp))
        preds[f"p_{name}"] = pp.tolist()
        tag = "  <-- marker ablation" if name == "corners" else ""
        print(f"  Perturbation {name:<10} AUC {res[f'auc_{name}']:.4f}  "
              f"({res[f'auc_{name}'] - res['auc']:+.4f}){tag}")

    # ---- Grad-CAM gegen Bounding Boxes ---------------------------------
    cam_df = pd.DataFrame()
    if args.cam_n:
        print(f"\n  Grad-CAM on {args.cam_n} positive val images (CPU)...")
        boxes = load_boxes(args.csv)
        cam_res, cam_df = cam_vs_boxes(model, args.images, val_ids, boxes,
                                       args.size, args.cam_n, args.seed)
        res.update(cam_res)
        if cam_res:
            print(f"  Hit rate {cam_res['cam_hit']:.3f}  vs chance "
                  f"{cam_res['cam_area_baseline']:.3f}  "
                  f"(margin {cam_res['cam_hit_lift']:+.3f})")
            print(f"  Mass     {cam_res['cam_mass']:.3f}  vs chance "
                  f"{cam_res['cam_area_baseline']:.3f}  "
                  f"(margin {cam_res['cam_mass_lift']:+.3f})")
            print("  Without the chance baseline the hit rate means nothing:")
            print("  the boxes cover a substantial part of the image.")

    # ---- Saving ---------------------------------------------------------
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds).to_csv(
        args.pred_dir / f"rsna_f{args.fold}_s{args.seed}.csv", index=False)
    # The selection predictions are written as well. Without them any later
    # question about the threshold can only be answered as an oracle (threshold
    # searched on the reporting set = too optimistic).
    pd.DataFrame({"patientId": sel_ids, "y": y_sel.tolist(),
                  "viewpos": vp_sel.tolist(), "p_sel": p_sel.tolist()}).to_csv(
        args.pred_dir / f"sel_f{args.fold}_s{args.seed}.csv", index=False)
    if not cam_df.empty:
        cam_df.to_csv(args.pred_dir / f"cam_f{args.fold}_s{args.seed}.csv", index=False)
    # Already on disk from the epoch loop; rewritten here only so the file is
    # certainly the selected state even if the loop never improved.
    torch.save(best_state, ckpt)

    # Read, merge and rewrite instead of appending. A plain mode="a" writes
    # the values in the order of the CURRENT run under a header that came from
    # an earlier one. As soon as the metric set changes, and thr_AP,
    # sens_AP_strat and the rest belong to it, 49 values sit under 41 column
    # names. The file is then not broken in the sense of unreadable, but worse:
    # silently shifted.
    row = pd.DataFrame([res])
    if args.out.exists():
        old = pd.read_csv(args.out)
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(args.out, index=False)
    print(f"\nsaved: {args.out}, {args.pred_dir}/, {ckpt}")


if __name__ == "__main__":
    main()
