"""
Checks the non-torch logic of rsna_train.py: box geometry, corner mask,
stratified metrics. Needs no GPU and no training.

The silent failure this file exists for: the Grad-CAM evaluation rescales the
bounding boxes from the 1024 DICOM grid to the 224 model grid. A scale error
does NOT show up in the finished number, because a hit rate of 0.4 looks
plausible whether or not the boxes sit in the right place. So the geometry is
checked here against cases worked out by hand.

  python test_rsna_train.py
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

FAILED = []


def check(name: str, cond: bool, info: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FAILED.append(name)


def _stub_torch() -> None:
    """Stubs ONLY when torch is genuinely missing (CI without a GPU stack).

    Stubbing on the mere absence of `torch` from `sys.modules` wrecks every
    environment where torch IS installed: on import, scipy asks
    array_api_compat for `torch.Tensor`, finds nothing on the empty module and
    raises an AttributeError in the middle of the sklearn import. So the
    import is tried for real first and only then replaced, and the replacement
    carries `Tensor`.
    """
    try:
        import torch            # noqa: F401
        import torchvision      # noqa: F401
        return
    except ImportError:
        pass
    for name in ["torch", "torch.nn", "torch.utils", "torch.utils.data",
                 "torchvision", "torchvision.transforms", "torchvision.models"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["torch"].Tensor = type("Tensor", (), {})
    sys.modules["torch"].no_grad = lambda: (lambda f: f)
    sys.modules["torch"].device = lambda s: types.SimpleNamespace(type=s)
    sys.modules["torch.utils.data"].Dataset = object
    sys.modules["torch.utils.data"].DataLoader = object
    sys.modules["torchvision.models"].resnet18 = None
    sys.modules["torchvision.models"].ResNet18_Weights = None
    sys.modules["torchvision"].transforms = sys.modules["torchvision.transforms"]


_stub_torch()

import _repo_path  # noqa: F401  (sets sys.path)
from rsna_train import (BOX_SPACE, MaskCorners, load_boxes,  # noqa: E402
                        stratified_scores, youden_threshold)


def box_mask(boxes, size):
    """The same geometry as in cam_vs_boxes, rebuilt here in isolation."""
    s = size / BOX_SPACE
    mask = np.zeros((size, size), bool)
    for bx, by, bw, bh in boxes:
        x0, y0 = int(bx * s), int(by * s)
        x1, y1 = int((bx + bw) * s), int((by + bh) * s)
        mask[max(y0, 0):y1, max(x0, 0):x1] = True
    return mask


def test_box_geometry() -> None:
    print("\nBox geometry 1024 -> 224")
    # box across the whole image -> mask fully set
    m = box_mask([(0, 0, 1024, 1024)], 224)
    check("full image box covers everything", m.all(), f"{m.mean():.3f}")

    # bottom right quarter: x,y = 512, w,h = 512
    m = box_mask([(512, 512, 512, 512)], 224)
    check("quarter box covers ~25 %", abs(m.mean() - 0.25) < 0.02, f"{m.mean():.3f}")
    check("quarter box sits bottom right",
          m[200, 200] and not m[20, 20] and not m[20, 200] and not m[200, 20])

    # x is the COLUMN, y the ROW. Classic transposition error.
    m = box_mask([(0, 512, 1024, 512)], 224)     # full width, lower half
    check("x=column, y=row not transposed",
          m[200, 20] and m[200, 200] and not m[20, 20],
          "lower half set, upper half not")

    # two separate boxes, typically one per lung
    m = box_mask([(100, 300, 250, 400), (650, 300, 250, 400)], 224)
    check("two boxes give two regions", 0.15 < m.mean() < 0.25, f"{m.mean():.3f}")

    # box runs past the edge -> must not wrap around
    m = box_mask([(900, 900, 400, 400)], 224)
    check("box past the edge does not wrap", m[-1, -1] and not m[0, 0])


def test_mask_corners() -> None:
    print("\nCorner mask")
    from PIL import Image
    # Background brightness deliberately != 0, otherwise "median == 0" holds
    # trivially and the test would wave a fill with zeros through.
    a = np.full((100, 100), 80, np.uint8)
    a[40:60, 40:60] = 200            # bright spot in the centre
    a[:10, :10] = 255                # "marker" in the corner
    out = np.asarray(MaskCorners(frac=0.18)(Image.fromarray(a)))
    check("corner overwritten", out[:18, :18].max() < 255, f"max {out[:18, :18].max()}")
    check("centre untouched", out[50, 50] == 200)
    check("fill value is the median, not 0", out[0, 0] == int(np.median(a)),
          f"{out[0, 0]} vs median {int(np.median(a))}")
    check("size unchanged", out.shape == a.shape)


def test_stratified() -> None:
    print("\nStratified metrics")
    rng = np.random.default_rng(0)
    n = 600
    vp = np.array(["AP"] * (n // 2) + ["PA"] * (n // 2))
    y = rng.integers(0, 2, n).astype(float)
    # a score that knows ONLY the projection: informative overall, blind
    # within each stratum
    y[vp == "AP"] = (rng.random((vp == "AP").sum()) < 0.7).astype(float)
    y[vp == "PA"] = (rng.random((vp == "PA").sum()) < 0.15).astype(float)
    p_view = (vp == "AP").astype(float) + rng.normal(0, 1e-6, n)
    r = stratified_scores(y, p_view, vp, thr=0.5)
    check("projection-only score is ~0.5 within each stratum",
          abs(r["auc_AP"] - 0.5) < 0.05 and abs(r["auc_PA"] - 0.5) < 0.05,
          f"AP {r['auc_AP']:.3f} PA {r['auc_PA']:.3f}")
    check("stratified AUC lies between the strata",
          min(r["auc_AP"], r["auc_PA"]) <= r["auc_stratified"] <= max(r["auc_AP"], r["auc_PA"]))
    check("n per stratum is right", r["n_AP"] == 300 and r["n_PA"] == 300)

    # a real score -> both strata clearly above 0.5
    p_real = y * 0.6 + rng.random(n) * 0.4
    r2 = stratified_scores(y, p_real, vp, thr=0.5)
    check("informative score above 0.8 in each stratum",
          r2["auc_AP"] > 0.8 and r2["auc_PA"] > 0.8,
          f"AP {r2['auc_AP']:.3f} PA {r2['auc_PA']:.3f}")


def test_threshold_per_view() -> None:
    """Does the per-stratum threshold close the sensitivity gap?

    Rebuilt here is the situation from the first run: the same discrimination
    in both projections, but very unequal prevalence. Exactly then one fixed
    threshold ends up at different sensitivities in AP and PA, and nothing in
    the overall numbers says so.
    """
    print("\nThreshold per projection")
    rng = np.random.default_rng(0)

    def block(n, pos_rate, view):
        y = (rng.random(n) < pos_rate).astype(float)
        # identical separation in both blocks, only shifted: the AP block
        # sits higher overall, as it does with the real model
        shift = 0.25 if view == "AP" else 0.0
        p = np.clip(y * 0.45 + rng.normal(0.25, 0.12, n) + shift, 0, 1)
        return y, p, np.full(n, view)

    ya, pa, va = block(2000, 0.383, "AP")
    yp, pp, vp_ = block(2000, 0.093, "PA")
    y = np.r_[ya, yp]; p = np.r_[pa, pp]; vp = np.r_[va, vp_]

    thr_g = youden_threshold(y, p)
    thr_v = {v: youden_threshold(y[vp == v], p[vp == v]) for v in ("AP", "PA")}
    r = stratified_scores(y, p, vp, thr_g, thr_v)

    check("both strata discriminate about equally well",
          abs(r["auc_AP"] - r["auc_PA"]) < 0.06,
          f"AP {r['auc_AP']:.3f} PA {r['auc_PA']:.3f}")
    check("global threshold produces a sensitivity gap", r["sens_gap"] > 0.15,
          f"{r['sens_gap']:.3f}")
    check("per-stratum threshold closes most of it",
          r["sens_gap_strat"] < r["sens_gap"] / 2,
          f"{r['sens_gap']:.3f} -> {r['sens_gap_strat']:.3f}")
    check("both thresholds are reported",
          "thr_AP" in r and "thr_PA" in r and r["thr_AP"] != r["thr_PA"])
    check("global values are kept",
          "sens_AP" in r and "sens_AP_strat" in r,
          "both variants sit side by side")

    # nothing may break when no per-stratum thresholds are passed
    r2 = stratified_scores(y, p, vp, thr_g)
    check("still runs without thr_by_view",
          "sens_gap" in r2 and "sens_gap_strat" not in r2)


def test_boxes_and_threshold() -> None:
    print("\nBox CSV and Youden threshold")
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        pd.DataFrame([
            {"patientId": "a", "x": np.nan, "y": np.nan, "width": np.nan,
             "height": np.nan, "Target": 0},
            {"patientId": "c", "x": 10, "y": 20, "width": 30, "height": 40, "Target": 1},
            {"patientId": "c", "x": 50, "y": 60, "width": 70, "height": 80, "Target": 1},
        ]).to_csv(d / "stage_2_train_labels.csv", index=False)
        b = load_boxes(d)
        check("only positives have boxes", set(b) == {"c"}, str(sorted(b)))
        check("both boxes captured", len(b["c"]) == 2)
        check("order is x,y,w,h", b["c"][0] == (10.0, 20.0, 30.0, 40.0))

    y = np.array([0, 0, 0, 1, 1, 1.0])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    check("perfectly separable -> threshold in the positive range",
          youden_threshold(y, p) == 0.7, str(youden_threshold(y, p)))


def test_bce_from_probs() -> None:
    """The loss behind the learning curve has to be the one used in training.

    `predict` returns probabilities, `BCEWithLogitsLoss` works on logits. The
    back transformation in `bce_from_probs` is therefore checked directly
    against torch. If it is only approximately right, the learning curve opens
    a gap between training and selection loss that does not exist, and one
    diagnoses overfitting that is in truth a bookkeeping convention.
    """
    print("\nLoss for the per-epoch history")
    from rsna_train import bce_from_probs

    rng = np.random.default_rng(0)
    y = (rng.random(500) > 0.7).astype(np.float64)
    logit = rng.normal(0, 2, 500)
    p = 1 / (1 + np.exp(-logit))

    try:
        import torch
        import torch.nn as nn
        for pw in (1.0, 3.44):
            ref = float(nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))(
                torch.tensor(logit), torch.tensor(y)))
            got = bce_from_probs(y, p, pw)
            check(f"matches BCEWithLogitsLoss (pos_weight={pw})",
                  abs(ref - got) < 1e-6, f"torch {ref:.6f} vs own {got:.6f}")
    except ImportError:
        check("torch comparison skipped (no torch)", True)

    sat = bce_from_probs(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    check("saturated prediction does not give infinity", np.isfinite(sat),
          f"{sat:.2e}")
    check("perfect prediction has almost no loss", sat < 1e-5, f"{sat:.2e}")

    schlecht = bce_from_probs(np.array([0.0, 1.0]), np.array([1.0, 0.0]))
    check("inverted prediction is clearly worse", schlecht > sat + 1.0,
          f"{schlecht:.2f} against {sat:.2e}")

    check("pos_weight weights the positives",
          abs(bce_from_probs(np.array([1.0]), np.array([0.3]), 2.0)
              - 2 * bce_from_probs(np.array([1.0]), np.array([0.3]), 1.0)) < 1e-12)
    check("and leaves the negatives untouched",
          abs(bce_from_probs(np.array([0.0]), np.array([0.3]), 2.0)
              - bce_from_probs(np.array([0.0]), np.array([0.3]), 1.0)) < 1e-12)


if __name__ == "__main__":
    test_box_geometry()
    test_mask_corners()
    test_stratified()
    test_threshold_per_view()
    test_boxes_and_threshold()
    test_bce_from_probs()
    print("\n" + ("ALL TESTS PASSED" if not FAILED
                  else f"{len(FAILED)} FAILED: {FAILED}"))
    raise SystemExit(1 if FAILED else 0)
