"""
Phase 1: the instrument that measures localisation, built before it is used.

WHY A NEW INSTRUMENT
--------------------
Until now localisation was the pointing game: does the maximum of the heat map
fall inside an annotated box, measured against `area`, the share of the image
the boxes cover. Two things are wrong with that pair, and neither can be fixed
by rescaling the numbers afterwards.

The chance value is too weak. `area` is the chance of a pointer that lands
anywhere with equal probability, including the image border and the air beside
the patient. Boxes, however, sit in the lung fields, and the lung fields sit in
roughly the same place in every chest radiograph. A method that has memorised
nothing but the usual location beats that chance value comfortably. The factor
of 4.6 reported so far therefore measures anatomy to an unknown degree, not
pathology.

The scale depends on the variant. In the crop the thorax fills the frame and
the boxes cover 16 instead of 12 percent. "Lift over its own chance value" does
not repair this: at 12 percent the lift can reach 0.88, at 16 percent only
0.84. The factor has the same problem with the opposite sign, it punishes large
chance values.

WHAT THIS MODULE PROVIDES
-------------------------
1. `location_prior`: every box of a fold's TRAINING part drawn into one grid and
   averaged. The result is one map, the same for every image, that knows
   nothing except where opacities usually sit. It is scored with the same
   measures as a real heat map and stands beside every localisation number from
   now on. Built per fold from the training part, never from all images, or it
   would carry validation information.
2. `lung_heat`: the existing U-Net mask, smoothed, as a heat map. "Point at the
   lung" is the trivial solution of the task and deserves a number.
3. `point_auc`: area under the curve over PIXELS. For every pixel it is known
   whether it lies inside a box, and the map supplies a value there. The AUC
   answers: draw one pixel inside a box and one outside, how often is the map
   higher inside. Its chance value is exactly 0.5, independent of box area and
   of cropping. Computed twice, over the whole image and restricted to the lung
   mask, because outside the lung "no box here" is trivially right and a large
   trivial area flatters the number.
4. `uncrop`: maps produced by a crop model are placed back into original
   coordinates with the stored crop parameters, so that both arms see the same
   boxes with the same area share.

PRIMARY IS THE POINT AUC INSIDE THE LUNG MASK. Hit rate and mass are still
reported so the older numbers stay connected, but they are secondary.

One limitation belongs in the text: the point AUC is not entirely blind to grid
spacing either, a finer map can enclose the box more tightly. It is no longer
inflated mechanically the way mass is, and its chance value no longer moves.

THE GATE
--------
The point AUC of the location prior has to sit clearly above 0.5. If it does
not, either the prior is built wrong or box position is less stereotyped than
assumed, and that has to be settled before it serves as a null line.

CLI, from the repository root:
  python rsna\\befunde\\rsna_lokalisation.py tor
  python rsna\\befunde\\rsna_lokalisation.py tor --folds 0 --limit 100
  python rsna\\befunde\\rsna_lokalisation.py rauchtest
"""

from __future__ import annotations

import _repo_path  # noqa: F401  (puts the neighbouring folders on sys.path)

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REF_SIZE = 224            # the fixed reference grid every map is scaled to
BOX_SPACE = 1024          # boxes are given in the original DICOM grid
LUNG_SIGMA_FRAC = 0.05    # smoothing of the lung map, as a fraction of the side
PRIOR_AUC_GATE = 0.60     # below this the prior is not usable as a null line


# --------------------------------------------------------------------------
# Geometry on the reference grid
# --------------------------------------------------------------------------

def box_mask(boxes, size: int = REF_SIZE, box_space: int = BOX_SPACE) -> np.ndarray:
    """Bounding boxes from the DICOM grid into the reference grid.

    Same truncation as `rsna_train.cam_vs_boxes`, on purpose: the new numbers
    have to stay comparable with the old ones down to the last pixel.
    """
    s = size / box_space
    m = np.zeros((size, size), bool)
    for bx, by, bw, bh in boxes:
        y0, y1 = max(int(by * s), 0), int((by + bh) * s)
        x0, x1 = max(int(bx * s), 0), int((bx + bw) * s)
        m[y0:y1, x0:x1] = True
    return m


def to_reference(heat: np.ndarray, size: int = REF_SIZE) -> np.ndarray:
    """Any map to the fixed reference grid, bilinear.

    Grad-CAM comes back at 7 by 7 for a 224 pixel input and at 10 by 10 for
    320. Without this step the two would be compared on different grids, and
    the finer one wins mechanically because it can enclose the box more
    tightly. That is exactly the artefact that made the mass comparison
    between 224 and 320 worthless.
    """
    a = np.asarray(heat, dtype=np.float32)
    if a.shape == (size, size):
        return a
    return np.asarray(Image.fromarray(a, mode="F").resize((size, size),
                                                          Image.BILINEAR),
                      dtype=np.float32)


def uncrop(heat: np.ndarray, top: float, left: float, side: float,
           size: int = REF_SIZE, fill: float | None = None) -> np.ndarray:
    """A map from crop space back into original coordinates.

    `top`, `left` and `side` are the fractions stored in `crop_params.csv`:
    the crop is the square that starts at (left, top) and has edge length
    `side`, all relative to the original image.

    Outside the crop the crop model produced nothing, and that is not the same
    as "there is nothing". The area is filled with the minimum of the map,
    which is the honest reading "no evidence available here": a box that the
    crop cut away counts as a miss, which it is. The share of the reference
    grid this affects is 1 - side squared, and it is reported.
    """
    a = to_reference(heat, size)
    k = max(int(round(side * size)), 1)
    y0 = min(max(int(round(top * size)), 0), size - 1)
    x0 = min(max(int(round(left * size)), 0), size - 1)
    k = min(k, size - y0, size - x0)
    inner = np.asarray(Image.fromarray(a, mode="F").resize((k, k), Image.BILINEAR),
                       dtype=np.float32)
    out = np.full((size, size), float(a.min()) if fill is None else float(fill),
                  dtype=np.float32)
    out[y0:y0 + k, x0:x0 + k] = inner
    return out


def location_prior(boxes: dict, ids, size: int = REF_SIZE) -> np.ndarray:
    """Every box of these images drawn into one grid and averaged.

    Pass the TRAINING ids of the fold. Passing all ids would put validation
    boxes into the null line the validation is measured against.
    """
    acc = np.zeros((size, size), dtype=np.float64)
    n = 0
    for pid in ids:
        b = boxes.get(pid)
        if not b:
            continue
        acc += box_mask(b, size)
        n += 1
    if n == 0:
        raise ValueError("no image with boxes, the prior would be empty")
    return (acc / n).astype(np.float32)


def lung_heat(lung: np.ndarray, sigma_frac: float = LUNG_SIGMA_FRAC) -> np.ndarray:
    """The lung mask as a heat map, smoothed.

    Smoothed for a reason: a raw binary mask has thousands of pixels sharing
    the maximum, so `argmax` would pick whichever one numpy reaches first and
    the hit rate of this baseline would be an artefact of array order. After
    smoothing the maximum sits in the middle of the largest lung area, which
    is what "point at the lung" means.
    """
    a = np.asarray(lung, dtype=np.float32)
    sigma = max(sigma_frac * a.shape[0], 0.5)
    try:
        import cv2
        return cv2.GaussianBlur(a, (0, 0), sigmaX=sigma, sigmaY=sigma)
    except ImportError:
        r = int(3 * sigma)
        x = np.arange(-r, r + 1, dtype=np.float32)
        k = np.exp(-0.5 * (x / sigma) ** 2)
        k /= k.sum()
        pad = np.pad(a, r, mode="edge")
        tmp = np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 1, pad)
        return np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 0, tmp)


# --------------------------------------------------------------------------
# The measure
# --------------------------------------------------------------------------

def point_auc(heat: np.ndarray, box: np.ndarray,
              valid: np.ndarray | None = None) -> float:
    """Area under the curve over pixels, with mid ranks for ties.

    Positive class are the pixels inside a box, negative class the rest. The
    chance value is exactly 0.5 whatever the box area is, which is the whole
    reason for switching to it.

    Mid ranks matter here more than usual. A Grad-CAM map upsampled from 7 by 7
    has large flat plateaus, and a crop map carries a constant fill outside the
    crop. Ranking those ties as if they were ordered would invent information
    that is not in the map.
    """
    h = np.asarray(heat, dtype=np.float64).ravel()
    b = np.asarray(box, dtype=bool).ravel()
    if valid is not None:
        v = np.asarray(valid, dtype=bool).ravel()
        h, b = h[v], b[v]
    n1 = int(b.sum())
    n0 = int(b.size - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")

    order = np.argsort(h, kind="mergesort")
    hs = h[order]
    new = np.empty(hs.size, dtype=bool)
    new[0] = True
    np.not_equal(hs[1:], hs[:-1], out=new[1:])
    grp = np.cumsum(new) - 1
    counts = np.bincount(grp)
    ends = np.cumsum(counts)
    starts = ends - counts
    mid = 0.5 * (starts + 1 + ends)          # average rank inside each tie group
    ranks = np.empty(hs.size, dtype=np.float64)
    ranks[order] = mid[grp]
    r1 = float(ranks[b].sum())
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def evaluate_map(heat: np.ndarray, box: np.ndarray,
                 lung: np.ndarray | None = None) -> dict:
    """All localisation measures for one image on the reference grid.

    Primary is `point_auc_lung`. `hit` and `mass` are carried along so the
    numbers reported so far stay connected, and they keep their own chance
    value `area` beside them, because on their own they mean nothing.
    """
    h = np.clip(np.asarray(heat, dtype=np.float64), 0, None)
    b = np.asarray(box, dtype=bool)
    total = float(h.sum())
    degenerate = bool(total <= 0)

    out = {
        "point_auc": point_auc(h, b),
        "point_auc_lung": float("nan"),
        "hit": False,
        "mass": 0.0,
        "area": float(b.mean()),
        "degenerate": degenerate,
        "lung_area": float("nan"),
        "box_in_lung": float("nan"),
    }
    if not degenerate:
        yx = np.unravel_index(int(np.argmax(h)), h.shape)
        out["hit"] = bool(b[yx])
        out["mass"] = float(h[b].sum() / total)

    if lung is not None:
        l = np.asarray(lung, dtype=bool)
        out["lung_area"] = float(l.mean())
        out["box_in_lung"] = float((b & l).sum() / b.sum()) if b.any() else float("nan")
        out["point_auc_lung"] = point_auc(h, b, l)
    return out


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_boxes(csv_dir: Path) -> dict:
    """Boxes per patient, independent of rsna_train so this stays torch free."""
    df = pd.read_csv(Path(csv_dir) / "stage_2_train_labels.csv")
    df = df[df["Target"] == 1].dropna(subset=["x", "y", "width", "height"])
    out: dict = {}
    for pid, x, y, w, h in df[["patientId", "x", "y", "width", "height"]].values:
        out.setdefault(str(pid), []).append((float(x), float(y), float(w), float(h)))
    return out


def load_lung(masks: Path, pid: str, size: int = REF_SIZE) -> np.ndarray | None:
    p = Path(masks) / f"{pid}.png"
    if not p.exists():
        return None
    m = np.array(Image.open(p).convert("L"))
    if m.shape != (size, size):
        raise ValueError(f"{p}: mask is {m.shape}, expected ({size}, {size}).")
    return m > 127


# --------------------------------------------------------------------------
# The gate: are the two null lines usable?
# --------------------------------------------------------------------------

def run_gate(args) -> int:
    sp = json.loads(Path(args.splits).read_text())
    boxes = load_boxes(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_fold, priors = [], {}
    t0 = time.time()

    # One file per fold, and a fold that is already on disk is not recomputed.
    # The run is therefore resumable, which matters because the sandbox this is
    # driven from cuts off long calls.
    for fold in args.folds:
        done = out_dir / f"baselines_f{fold}.csv"
        prior_npy = out_dir / f"prior_f{fold}.npy"
        if done.exists() and prior_npy.exists() and not args.force:
            print(f"\nFold {fold}: {done.name} is already there, skipped "
                  f"(--force recomputes)")
            continue

        train_ids = sp["folds"][fold]["train"]
        val_ids = sp["folds"][fold]["val"]
        prior = location_prior(boxes, train_ids)
        np.save(prior_npy, prior)

        pos = [i for i in val_ids if i in boxes]
        if args.limit:
            pos = pos[:args.limit]
        print(f"\nFold {fold}: prior from {sum(1 for i in train_ids if i in boxes)} "
              f"training images, scored on {len(pos)} validation images with boxes")

        rows, no_mask = [], 0
        for j, pid in enumerate(pos, 1):
            lung = load_lung(args.masks, pid)
            if lung is None:
                no_mask += 1
                continue
            b = box_mask(boxes[pid])
            lheat = lung_heat(lung)
            for name, heat in (("Lagepriore", prior), ("Lungenkarte", lheat)):
                r = evaluate_map(heat, b, lung)
                r.update({"fold": fold, "patientId": pid, "map": name})
                rows.append(r)
            if j % 250 == 0:
                print(f"    {j}/{len(pos)}   [{(time.time() - t0) / 60:.1f} min]")
        if no_mask:
            print(f"  {no_mask} images without a lung mask skipped")
        if rows:
            pd.DataFrame(rows).to_csv(done, index=False)

    have = sorted(out_dir.glob("baselines_f*.csv"))
    if not have:
        print("nothing measured")
        return 2
    d = pd.concat([pd.read_csv(f) for f in have], ignore_index=True)
    d.to_csv(out_dir / "baselines_per_image.csv", index=False)
    for f in sorted(out_dir.glob("prior_f*.npy")):
        priors[int(f.stem.split("_f")[1])] = np.load(f)

    for (fold, name), g in d.groupby(["fold", "map"]):
        per_fold.append({
            "fold": fold, "map": name, "n": len(g),
            "point_auc": g["point_auc"].mean(),
            "point_auc_lung": g["point_auc_lung"].mean(),
            "hit": g["hit"].mean(), "mass": g["mass"].mean(),
            "area": g["area"].mean(), "lung_area": g["lung_area"].mean(),
            "box_in_lung": g["box_in_lung"].mean(),
            "n_auc_lung_undefined": int(g["point_auc_lung"].isna().sum()),
        })
    s = pd.DataFrame(per_fold).sort_values(["map", "fold"])
    s.to_csv(out_dir / "baselines_summary.csv", index=False)

    print("\n" + "=" * 78)
    print("THE TWO NULL LINES, per fold")
    print("=" * 78)
    print(s.round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("MEAN OVER FOLDS, chance is 0.5 for both point AUCs")
    print("=" * 78)
    for name in ("Lagepriore", "Lungenkarte"):
        g = s[s["map"] == name]
        if g.empty:
            continue
        for col, label in (("point_auc", "point AUC, whole image"),
                           ("point_auc_lung", "point AUC inside the lung")):
            v = g[col].to_numpy(dtype=float)
            sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
            print(f"  {name:<12} {label:<28} {v.mean():.4f} +- {sd:.4f}")
        print(f"  {name:<12} {'hit rate':<28} {g['hit'].mean():.4f}   "
              f"chance (box area) {g['area'].mean():.4f}")

    if len(priors) > 1:
        f = sorted(priors)
        c = np.corrcoef(np.stack([priors[k].ravel() for k in f]))
        off = c[np.triu_indices(len(f), 1)]
        print(f"\n  Priors of the folds correlate {off.min():.4f} to {off.max():.4f} "
              f"with each other.")
        print("  Close to 1 is expected and is the point: the prior is anatomy,")
        print("  not fold noise. It still gets built per fold, because only that")
        print("  keeps validation boxes out of the null line.")

    lag = s[s["map"] == "Lagepriore"]["point_auc_lung"].mean()
    print("\n" + "=" * 78)
    print("GATE")
    print("=" * 78)
    if not np.isfinite(lag):
        print("  FAILED: the point AUC of the prior inside the lung is not defined.")
        return 1
    if lag < PRIOR_AUC_GATE:
        print(f"  FAILED: prior at {lag:.4f}, below {PRIOR_AUC_GATE:.2f}.")
        print("  Either the prior is built wrong or box position is less")
        print("  stereotyped than assumed. Settle that before it serves as a")
        print("  null line, do not start phase 2 on it.")
        return 1
    print(f"  passed: the prior reaches {lag:.4f} inside the lung against a chance")
    print("  value of 0.5, knowing nothing but where opacities usually sit.")
    print("  Every model number from now on is reported beside this one.")
    print(f"\nsaved: {out_dir}/baselines_per_image.csv, "
          f"{out_dir}/baselines_summary.csv, prior_f*.npy")
    return 0


# --------------------------------------------------------------------------
# The smoke test: does the back projection land where it should?
# --------------------------------------------------------------------------

def run_smoke(args) -> int:
    """Sends a known map through crop space and back, and checks it returns.

    A back projection that is off by a few percent shows up in no number and
    ruins every comparison, so it is tested against ground truth rather than
    looked at. The known map is the box map itself: drawn in crop coordinates,
    sent back through `uncrop`, it has to land on the boxes in original
    coordinates. Perfect agreement is impossible, the map passes through two
    resamplings, so the criterion is a point AUC near 1 and a peak inside the
    original box.

    The overlay PNG is the second half of the test, for the eye: the thorax has
    to sit where the thorax sits.
    """
    cp = pd.read_csv(args.crop_params).set_index("patientId")
    boxes = load_boxes(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    have = [p for p in cp.index.astype(str) if p in boxes]
    rng = np.random.default_rng(args.seed)
    pick = list(rng.choice(have, min(args.n, len(have)), replace=False))

    rows = []
    for pid in pick:
        r = cp.loc[pid]
        if not bool(r["ok"]):
            continue
        top, left, side = float(r["top"]), float(r["left"]), float(r["side"])

        # The boxes as the crop model sees them: shifted and scaled into the
        # crop, which is what rsna_make_crops.py does to the annotations.
        s = REF_SIZE / BOX_SPACE
        crop_boxes = []
        for bx, by, bw, bh in boxes[pid]:
            crop_boxes.append(((bx * s - left * REF_SIZE) / side,
                               (by * s - top * REF_SIZE) / side,
                               bw * s / side, bh * s / side))
        in_crop = box_mask(crop_boxes, REF_SIZE, box_space=REF_SIZE).astype(np.float32)
        back = uncrop(in_crop, top, left, side)
        truth = box_mask(boxes[pid], REF_SIZE)

        yx = np.unravel_index(int(np.argmax(back)), back.shape)
        rows.append({
            "patientId": pid, "side": side, "outside_frac": 1 - side ** 2,
            "point_auc_back": point_auc(back, truth),
            "peak_in_true_box": bool(truth[yx]),
            "n_boxes": len(boxes[pid]),
        })

    if not rows:
        print("no usable image")
        return 2
    d = pd.DataFrame(rows)
    d.to_csv(out_dir / "uncrop_smoke.csv", index=False)

    auc = d["point_auc_back"].mean()
    peak = d["peak_in_true_box"].mean()
    print("\n" + "=" * 78)
    print(f"BACK PROJECTION, {len(d)} images")
    print("=" * 78)
    print(f"  point AUC of the returned box map against the true boxes  {auc:.4f}")
    print(f"  peak inside the true box                                  {peak:.4f}")
    print(f"  worst single image                                        "
          f"{d['point_auc_back'].min():.4f}")
    print(f"  share of the frame outside the crop, mean                 "
          f"{d['outside_frac'].mean():.4f}")
    ok = auc > 0.98 and peak > 0.95
    print(f"\n  {'passed' if ok else 'FAILED'}: a map drawn in crop coordinates "
          f"returns onto its own boxes.")
    if not ok:
        print("  The back projection is displaced. Do not compare crop maps with")
        print("  whole image maps until this is fixed.")

    # The overlay, for the eye.
    if args.overlay:
        pid = d.sort_values("side").iloc[0]["patientId"]
        r = cp.loc[pid]
        img = Image.open(Path(args.images) / f"{pid}.png").convert("L").resize(
            (REF_SIZE, REF_SIZE))
        base = np.asarray(img, dtype=np.float32) / 255.0
        frame = np.zeros((REF_SIZE, REF_SIZE), np.float32)
        k = int(round(float(r["side"]) * REF_SIZE))
        y0, x0 = int(round(float(r["top"]) * REF_SIZE)), int(round(float(r["left"]) * REF_SIZE))
        frame[y0:y0 + k, x0:x0 + k] = 1.0
        truth = box_mask(boxes[pid], REF_SIZE)
        rgb = np.stack([base, base, base], -1)
        rgb[..., 0] = np.where(frame > 0, np.clip(rgb[..., 0] + 0.25, 0, 1), rgb[..., 0])
        rgb[..., 1] = np.where(truth, np.clip(rgb[..., 1] + 0.35, 0, 1), rgb[..., 1])
        p = out_dir / "uncrop_overlay.png"
        Image.fromarray((rgb * 255).astype(np.uint8)).save(p)
        print(f"\n  overlay written: {p}  (red = crop frame, green = boxes, "
              f"image {pid})")
    return 0 if ok else 1


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("tor", help="build both null lines and check the gate")
    g.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    g.add_argument("--csv", type=Path, default=Path("data/rsna"))
    g.add_argument("--masks", type=Path, default=Path("data/rsna/masks224_dev"))
    g.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    g.add_argument("--limit", type=int, default=0,
                   help="images per fold, 0 = every validation image with boxes")
    g.add_argument("--force", action="store_true",
                   help="recompute folds that are already on disk")
    g.add_argument("--out-dir", type=Path, default=Path("predictions_lokalisation"))
    g.set_defaults(func=run_gate)

    s = sub.add_parser("rauchtest", help="check the crop back projection")
    s.add_argument("--csv", type=Path, default=Path("data/rsna"))
    s.add_argument("--images", type=Path, default=Path("data/rsna/png512"))
    s.add_argument("--crop-params", type=Path,
                   default=Path("predictions_rsna/crop_params.csv"))
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--overlay", action="store_true", default=True)
    s.add_argument("--out-dir", type=Path, default=Path("predictions_lokalisation"))
    s.set_defaults(func=run_smoke)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
