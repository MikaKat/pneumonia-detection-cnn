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

CLI:
  python rsna_train.py --fold 0 --epochs 8 --batch 16 --workers 0
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
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "directml", "cpu"])
    p.add_argument("--cam-n", type=int, default=300, help="0 = skip Grad-CAM")
    p.add_argument("--out", type=Path, default=Path("results_rsna.csv"))
    p.add_argument("--pred-dir", type=Path, default=Path("predictions_rsna"))
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

    tr = DataLoader(RsnaDataset(args.images, fit_ids, labels,
                               build_transforms(args.size, True)),
                    batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=pin, drop_last=True)
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
                "n_sel": len(sel_ids), "n_val": len(val_ids)})

    print(f"\n  AUC overall     {res['auc']:.4f}   (last epoch {auc_last:.4f}, "
          f"header baseline 0.729)")
    for v in ("AP", "PA"):
        if f"auc_{v}" in res:
            print(f"  AUC {v} only     {res[f'auc_{v}']:.4f}   "
                  f"(n={res[f'n_{v}']}, pos={res[f'pos_{v}']:.3f}, "
                  f"baseline ~0.556)")
    if "auc_stratified" in res:
        print(f"  AUC stratified  {res['auc_stratified']:.4f}  <-- the honest number")
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
    torch.save(best_state, f"checkpoints/rsna_f{args.fold}_s{args.seed}.pth")

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
    print(f"\nsaved: {args.out}, {args.pred_dir}/, "
          f"checkpoints/rsna_f{args.fold}_s{args.seed}.pth")


if __name__ == "__main__":
    main()
