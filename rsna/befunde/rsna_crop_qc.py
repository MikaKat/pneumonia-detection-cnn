"""
Visual check of the lung crop, separated by projection.

What it produces
----------------
A PNG with two blocks, AP on top and PA below. Each column shows one case
twice: the original 512 px film with the lung mask outlined and the crop window
drawn in, and underneath the resulting crop with the transformed bounding box.
A histogram of the magnification factor by projection sits at the bottom.

Why it exists
-------------
Every other step in this pipeline has a quality-control preview; the crop did
not, and that gap cost a full paired experiment. The adaptive crop lowered no
confounder. It moved one: the window is fitted to the lung mask, lungs look
smaller on supine AP films, so AP images were magnified about 7 % more than PA
images and the projection became readable from the scale of the anatomy
instead of from the framing. That is visible at a glance in a panel like this
and invisible in a single AUC.

The preview separates AP from PA for exactly that reason. A grid of mixed
cases looks fine; the systematic difference only shows up when the two
projections stand next to each other.

Reading the output
------------------
Look at the second row of each block, the crops themselves. If the thorax
fills the frame noticeably more in the AP block than in the PA block, the crop
is applying a projection-dependent magnification and any downstream confounder
measurement will pick that up. With a fixed side length the two blocks should
be indistinguishable in that respect, and only the position of the thorax
inside the frame should vary.

The histogram is the same statement as a number. One narrow peak shared by
both projections means the magnification is constant. Two offset peaks mean it
is not.

CLI:
  # adaptive crop, as used in the first comparison
  python rsna_crop_qc.py --params predictions_rsna/crop_params.csv

  # a fixed side length, before committing compute to it
  python rsna_crop_qc.py --fixed-side 0.85 --out qc/crop_qc_fixed085.png
"""

from __future__ import annotations

import _repo_path  # noqa: F401  (setzt sys.path fuer die Nachbarordner)

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BOX_SPACE = 1024


def window(row: pd.Series, fixed_side: float | None) -> tuple[float, float, float]:
    """The crop window as (top, left, side), all in [0,1].

    With a fixed side length the centre of the adaptive window is kept and only
    the size is replaced, then the window is pushed back inside the image. That
    is the same rule the crop itself uses, so the preview cannot disagree with
    what would be written to disk.
    """
    if fixed_side is None:
        return float(row.top), float(row.left), float(row.side)
    s = min(fixed_side, 1.0)
    cy = float(row.top) + float(row.side) / 2
    cx = float(row.left) + float(row.side) / 2
    top = min(max(cy - s / 2, 0.0), 1.0 - s)
    left = min(max(cx - s / 2, 0.0), 1.0 - s)
    return top, left, s


def pick(d: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """n positive cases per projection, drawn reproducibly.

    Positives only: without an annotated box there is nothing to check the
    transformed coordinates against.
    """
    rng = np.random.default_rng(seed)
    out = []
    for v in ("AP", "PA"):
        pool = d[(d.vp == v) & (d.target == 1)]
        if pool.empty:
            continue
        take = rng.choice(len(pool), min(n, len(pool)), replace=False)
        out.append(pool.iloc[sorted(take)])
    return pd.concat(out) if out else d.iloc[:0]


def draw(params: Path, images: Path, cache: Path, csv_dir: Path, out: Path,
         fixed_side: float | None, n: int, refine: str, dilate_px: int,
         seed: int, splits: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    from rsna_make_crops import boxes_by_id, transform_boxes
    from rsna_make_masks import load_raw_cache, refine_variant, unpack_masks

    d = pd.read_csv(params)
    d["patientId"] = d["patientId"].astype(str)
    # rsna_make_crops.py writes geometry only, so projection and label are
    # joined in from the split file rather than required in the parameter CSV.
    if "vp" not in d.columns or "target" not in d.columns:
        sp = json.loads(Path(splits).read_text())
        d["vp"] = d["patientId"].map(sp["viewpos"]).fillna("?")
        d["target"] = d["patientId"].map(
            {k: int(v) for k, v in sp["labels"].items()}).fillna(-1).astype(int)
    boxes = boxes_by_id(pd.read_csv(Path(csv_dir) / "stage_2_train_labels.csv"))
    ids, packed = load_raw_cache(cache)
    index = {p: i for i, p in enumerate(ids)}

    sel = pick(d, n, seed)
    cols = max(len(sel[sel.vp == "AP"]), len(sel[sel.vp == "PA"]))
    fig, axes = plt.subplots(5, cols, figsize=(2.4 * cols, 12.5))
    if cols == 1:
        axes = axes.reshape(5, 1)

    for block, v in enumerate(("AP", "PA")):
        rows = sel[sel.vp == v]
        for j in range(cols):
            a_src, a_crop = axes[2 * block, j], axes[2 * block + 1, j]
            for a in (a_src, a_crop):
                a.set_xticks([]); a.set_yticks([])
            if j >= len(rows):
                a_src.axis("off"); a_crop.axis("off")
                continue
            r = rows.iloc[j]
            pid = str(r.patientId)
            img = np.array(Image.open(Path(images) / f"{pid}.png").convert("L"))
            H, W = img.shape
            top, left, side = window(r, fixed_side)

            a_src.imshow(img, cmap="gray")
            if pid in index:
                m = refine_variant(unpack_masks(packed[index[pid]:index[pid] + 1])[0],
                                   refine, dilate_px)
                a_src.contour(np.kron(m, np.ones((H // m.shape[0], W // m.shape[1]))),
                              levels=[0.5], colors="tab:cyan", linewidths=0.8)
            a_src.add_patch(Rectangle((left * W, top * H), side * W, side * H,
                                      fill=False, color="tab:orange", lw=1.4))
            if j == 0:
                a_src.set_ylabel(f"{v}\noriginal", fontsize=9)

            s_px = max(int(round(side * H)), 1)
            y0 = min(int(round(top * H)), H - s_px)
            x0 = min(int(round(left * W)), W - s_px)
            patch = img[y0:y0 + s_px, x0:x0 + s_px]
            a_crop.imshow(patch, cmap="gray")
            ph, pw = patch.shape
            for bx, by, bw, bh in transform_boxes(boxes.get(pid, []),
                                                  (top, left, side))[0]:
                a_crop.add_patch(Rectangle((bx / BOX_SPACE * pw, by / BOX_SPACE * ph),
                                           bw / BOX_SPACE * pw, bh / BOX_SPACE * ph,
                                           fill=False, color="tab:red", lw=1.2))
            a_crop.set_xlabel(f"{1 / side:.2f}x", fontsize=8)
            if j == 0:
                a_crop.set_ylabel(f"{v}\ncrop", fontsize=9)

    # magnification by projection, the same statement as a number
    gs = axes[4, 0].get_gridspec()
    for a in axes[4, :]:
        a.remove()
    hist = fig.add_subplot(gs[4, :])
    e = d[d.vp.isin(["AP", "PA"])]
    for v, c in (("AP", "tab:red"), ("PA", "tab:blue")):
        z = np.array([1 / window(r, fixed_side)[2] for _, r in e[e.vp == v].iterrows()])
        hist.hist(z, bins=60, alpha=0.55, color=c, label=f"{v}  mean {z.mean():.3f}x")
    hist.set_xlabel("magnification (1 / side)")
    hist.set_ylabel("images")
    hist.legend()

    # The caption describes what was DRAWN, not what the caller passed.
    #
    # It used to read `fixed_side if fixed_side else "adaptive"`, which takes
    # the caption from a switch rather than from the data. Called with
    # `--params crop_params_fix080.csv` and no `--fixed-side`, the picture
    # showed a perfectly constant window and was titled "adaptive crop, side
    # fitted per image": the variant this project measured and rejected. Months
    # later, or in a portfolio, nothing in the file would correct that.
    #
    # Same lesson as `side_sd` in rsna_make_crops.py, and the same remedy: the
    # spread comes from max minus min, so a constant side gives exactly 0.0
    # with no floating point noise in between.
    seiten = np.array([window(r, fixed_side)[2] for _, r in d.iterrows()])
    fest = float(seiten.max() - seiten.min()) == 0.0
    title = (f"fixed side length {float(seiten[0]):.2f}" if fest
             else "adaptive crop, side fitted per image")
    fig.suptitle(f"Lung crop, {title}. Cyan: lung mask. Orange: crop window. "
                 f"Red: annotated box.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"written: {out}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", type=Path,
                   default=Path("predictions_rsna/crop_params.csv"))
    p.add_argument("--images", type=Path, default=Path("data/rsna/png512"))
    p.add_argument("--raw-cache", type=Path,
                   default=Path("data/rsna/unet_raw256.npz"))
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--splits", type=Path, default=Path("rsna_splits.json"),
                   help="where projection and label come from when the "
                        "parameter file does not carry them")
    p.add_argument("--out", type=Path, default=Path("qc/crop_qc.png"))
    p.add_argument("--fixed-side", type=float, default=None,
                   help="preview a constant side length instead of the fitted one")
    p.add_argument("--n", type=int, default=6, help="cases per projection")
    p.add_argument("--refine", default="hull")
    p.add_argument("--dilate-px", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    for f in (args.params, args.raw_cache):
        if not f.exists():
            print(f"ERROR: missing: {f}")
            return 2
    draw(args.params, args.images, args.raw_cache, args.csv, args.out,
         args.fixed_side, args.n, args.refine, args.dilate_px, args.seed,
         args.splits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
