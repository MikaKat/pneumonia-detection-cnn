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
from rsna_train import (BOX_SPACE, MaskCorners, balance_report,  # noqa: E402
                        effective_n, load_boxes, residual_view_label_auc,
                        stratified_scores, train_loader_kwargs,
                        view_balance_weights, youden_threshold)


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


def test_view_balance_weights() -> None:
    """The reweighting behind --balance-view.

    The silent failure this guards against: a weight scheme that looks
    plausible in the printed table but leaves part of the projection-label
    association standing, or that quietly also rebalances the classes and so
    corrects the imbalance twice once `pos_weight` is applied on top. Neither
    shows up in the training log. Both show up here.
    """
    print("\ntest_view_balance_weights")

    # The real cell counts of the development set, so the numbers in the module
    # header are checked and not just asserted there.
    vp = np.array(["AP"] * 10434 + ["PA"] * 12438)
    y = np.array([0] * 6436 + [1] * 3998 + [0] * 11282 + [1] * 1156)
    w = view_balance_weights(y, vp)
    cell = {(r["viewpos"], r["target"]): r["weight"]
            for r in balance_report(w, y, vp)}
    for (v, t), want in {("AP", 0): 1.256, ("AP", 1): 0.588,
                         ("PA", 0): 0.854, ("PA", 1): 2.424}.items():
        check(f"weight {v}/{t} matches the module header",
              abs(cell[(v, t)] - want) < 5e-3, f"{cell[(v, t)]:.3f}")

    # 1. The association is gone: weighted, every cell holds exactly the count
    #    expected under independence.
    n = len(y)
    worst = 0.0
    for v in ("AP", "PA"):
        for t in (0, 1):
            m = (vp == v) & (y == t)
            got = w[m].sum()
            want = (vp == v).sum() * (y == t).sum() / n
            worst = max(worst, abs(got - want))
    check("weighted cells equal the independence expectation", worst < 1e-6,
          f"largest deviation {worst:.2e}")

    # 2. Both marginals survive. This is what keeps pos_weight valid: if the
    #    weighting also shifted the prevalence, the class imbalance would be
    #    corrected once here and once in the loss.
    check("projection marginal unchanged",
          abs(w[vp == "AP"].sum() - (vp == "AP").sum()) < 1e-6)
    check("prevalence unchanged",
          abs(w[y == 1].sum() / w.sum() - (y == 1).mean()) < 1e-9,
          f"{w[y == 1].sum() / w.sum():.6f} against {(y == 1).mean():.6f}")

    # 3. Weights sum to n, so an epoch keeps its length.
    check("weights sum to n", abs(w.sum() - n) < 1e-6)

    # 4. The price tag, and that it is a price and not a gain.
    ne = effective_n(w)
    check("effective sample size below n", ne < n, f"{ne:.0f} of {n}")
    check("effective sample size as documented", abs(ne - 19698) < 5,
          f"{ne:.0f}")
    check("equal weights give back n exactly",
          abs(effective_n(np.ones(500)) - 500) < 1e-9)

    # 5. Degenerate inputs must be no-ops, not crashes. A fold that happens to
    #    hold one projection only would otherwise divide by zero.
    one_view = view_balance_weights(np.array([0, 1, 1, 0]),
                                    np.array(["AP"] * 4))
    check("single projection is a no-op", np.allclose(one_view, 1.0))
    one_class = view_balance_weights(np.zeros(4, int),
                                     np.array(["AP", "AP", "PA", "PA"]))
    check("single class is a no-op", np.allclose(one_class, 1.0))
    check("empty input returns empty",
          view_balance_weights(np.zeros(0, int), np.zeros(0, str)).size == 0)

    # 6. An unknown ViewPosition forms its own stratum instead of being
    #    silently merged into AP or PA.
    mixed_vp = np.array(["AP", "AP", "PA", "PA", "?", "?"])
    mixed_y = np.array([0, 1, 0, 1, 0, 1])
    wm = view_balance_weights(mixed_y, mixed_vp)
    check("unknown projection is its own stratum", np.allclose(wm, 1.0),
          "balanced input stays balanced")


def test_balance_strength() -> None:
    """The dial between baseline and full independence.

    The silent failure here: `w ** a` no longer sums to n, so without
    renormalisation the epoch would quietly change length with the setting and
    a dose-response curve would confound dose with training time. That is
    invisible in the printed weight table and visible here.
    """
    print("\ntest_balance_strength")
    vp = np.array(["AP"] * 10434 + ["PA"] * 12438)
    y = np.array([1] * 3998 + [0] * 6436 + [1] * 1156 + [0] * 11282)
    n = len(y)

    # The dose axis has to reproduce the two anchors of the project.
    check("untouched stream reproduces the documented 0.706",
          abs(residual_view_label_auc(y, vp) - 0.706) < 5e-4,
          f"{residual_view_label_auc(y, vp):.4f}")
    w1 = view_balance_weights(y, vp, 1.0)
    check("full strength gives exactly 0.500",
          abs(residual_view_label_auc(y, vp, w1) - 0.5) < 1e-9)

    w0 = view_balance_weights(y, vp, 0.0)
    check("strength 0 is the untouched baseline", np.allclose(w0, 1.0))
    check("strength 1 equals the default",
          np.allclose(w1, view_balance_weights(y, vp)))

    # Every setting must keep the epoch the same length.
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        w = view_balance_weights(y, vp, a)
        if abs(w.sum() - n) > 1e-6:
            check(f"weights sum to n at strength {a}", False, f"{w.sum():.1f}")
            break
    else:
        check("weights sum to n at every strength", True)

    # Monotone in both directions, which is what makes it a dose axis.
    res = [residual_view_label_auc(y, vp, view_balance_weights(y, vp, a))
           for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
    ne = [effective_n(view_balance_weights(y, vp, a))
          for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
    check("residual association falls monotonically",
          all(x > z for x, z in zip(res, res[1:])),
          " ".join(f"{r:.3f}" for r in res))
    check("effective sample size falls monotonically",
          all(x > z for x, z in zip(ne, ne[1:])),
          " ".join(f"{v:.0f}" for v in ne))
    check("half strength halves the association roughly",
          abs((res[2] - 0.5) / (res[0] - 0.5) - 0.5) < 0.10,
          f"{(res[2] - 0.5) / (res[0] - 0.5):.2f} of the original gap")


def test_balance_flag_is_inert_when_off() -> None:
    """With the flag off the run has to be the baseline, bit for bit.

    The paired comparison is only worth anything if the two runs differ in the
    tested quantity and in nothing else. So the off path must not compute
    weights, must not build a sampler and above all must not touch the random
    number generators, since a single extra draw would shift the whole shuffle
    order and with it the result.
    """
    print("\ntest_balance_flag_is_inert_when_off")
    y = np.array([0, 1, 0, 1, 1, 0])
    vp = np.array(["AP", "AP", "PA", "PA", "AP", "PA"])

    off = train_loader_kwargs(None, seed=0)
    check("off yields exactly shuffle=True", off == {"shuffle": True}, str(off))

    # `_stub_torch` leaves a placeholder module behind when torch is genuinely
    # missing, so a bare `import torch` succeeds and then fails on the first
    # attribute. Ask for what is actually needed instead.
    try:
        import torch
        torch.Generator
    except (ImportError, AttributeError):
        print("  skip  sampler path (torch not installed)")
        return

    # The RNG must be in the same state before and after the off call.
    torch.manual_seed(123)
    before = torch.randint(0, 2 ** 31, (1,)).item()
    torch.manual_seed(123)
    train_loader_kwargs(None, seed=0)
    after = torch.randint(0, 2 ** 31, (1,)).item()
    check("off does not advance the torch RNG", before == after)

    w = view_balance_weights(y, vp)
    on = train_loader_kwargs(w, seed=0)
    check("on yields a sampler and no shuffle",
          "sampler" in on and "shuffle" not in on, str(list(on)))
    check("sampler draws one epoch worth of images",
          on["sampler"].num_samples == len(y))
    check("sampler draws with replacement", on["sampler"].replacement)

    # Same seed, same draw. Otherwise a rerun is not a rerun.
    a = list(train_loader_kwargs(w, seed=7)["sampler"])
    b = list(train_loader_kwargs(w, seed=7)["sampler"])
    c = list(train_loader_kwargs(w, seed=8)["sampler"])
    check("same seed gives the same draw", a == b)
    check("a different seed gives a different draw", a != c)

    # THE regression test. train_loader_kwargs used to take (y, vp, strength)
    # and recompute the weights itself; main printed its table from one call
    # and built the sampler from another, and the sampler call was left without
    # the strength argument. A run announced strength 0.5 and trained at 1.0
    # for 74 minutes. The signature now takes the finished array, so the two
    # cannot come apart, and this checks that the array actually reaches the
    # sampler unchanged.
    w05 = view_balance_weights(y, vp, 0.5)
    s05 = train_loader_kwargs(w05, seed=0)["sampler"]
    s10 = train_loader_kwargs(view_balance_weights(y, vp, 1.0),
                              seed=0)["sampler"]
    check("the sampler carries the weights it was handed",
          np.allclose(np.asarray(s05.weights, dtype=float), w05))
    check("different strengths give different samplers",
          not np.allclose(np.asarray(s05.weights, dtype=float),
                          np.asarray(s10.weights, dtype=float)))
    check("and therefore a different draw",
          list(s05) != list(train_loader_kwargs(
              view_balance_weights(y, vp, 1.0), seed=0)["sampler"]))


if __name__ == "__main__":
    test_box_geometry()
    test_mask_corners()
    test_stratified()
    test_threshold_per_view()
    test_boxes_and_threshold()
    test_bce_from_probs()
    test_view_balance_weights()
    test_balance_strength()
    test_balance_flag_is_inert_when_off()
    print("\n" + ("ALL TESTS PASSED" if not FAILED
                  else f"{len(FAILED)} FAILED: {FAILED}"))
    raise SystemExit(1 if FAILED else 0)
