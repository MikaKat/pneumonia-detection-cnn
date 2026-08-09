"""
Checks the crop geometry of `rsna_make_crops.py` against cases computed by hand.

A scale factor or a sign error in the crop shows up in no result number. The
model trains on images cut slightly wrong and still reports a plausible AUC,
the probability that it ranks a random pneumonia case above a random negative
one. The Grad-CAM hit rate behaves the same way. It counts how often the image
region the model reacted to falls inside the annotated box, and a middling
value there reads the same whether or not the boxes were moved along with the
crop. Such an error shows itself only under a recomputation by hand, so every
case below is computed by hand.

  python test_rsna_crops.py
"""

from __future__ import annotations

import numpy as np

import _repo_path  # noqa: F401  (sets sys.path)

import rsna_make_crops as mc

FAILED: list[str] = []


def check(name: str, cond: bool, info: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FAILED.append(name)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


def test_mask_bbox() -> None:
    print("\nbounding rectangle of the mask")
    m = np.zeros((100, 100), bool)
    m[20:40, 30:70] = True
    bb = mc.mask_bbox(m)
    check("oben", close(bb[0], 0.20), str(bb[0]))
    check("links", close(bb[1], 0.30), str(bb[1]))
    check("unten", close(bb[2], 0.40), str(bb[2]))
    check("rechts", close(bb[3], 0.70), str(bb[3]))
    check("empty mask -> None", mc.mask_bbox(np.zeros((10, 10), bool)) is None)

    einzeln = np.zeros((10, 10), bool)
    einzeln[5, 5] = True
    bb = mc.mask_bbox(einzeln)
    check("single pixel gives a rectangle 1/10 wide",
          close(bb[2] - bb[0], 0.1) and close(bb[3] - bb[1], 0.1))


def test_square_crop_ist_quadratisch() -> None:
    print("\ncrop window is square and stays inside the image")
    # wide rectangle in the middle of the image
    top, left, side = mc.square_crop((0.40, 0.20, 0.60, 0.80), pad=0.0)
    check("side length = longer edge", close(side, 0.60), str(side))
    check("horizontal position unchanged", close(left, 0.20), str(left))
    check("grows vertically about the centre", close(top, 0.20), str(top))
    check("stays inside the image", top >= 0 and left >= 0
          and top + side <= 1 + 1e-12 and left + side <= 1 + 1e-12)

    # tall rectangle
    top, left, side = mc.square_crop((0.10, 0.45, 0.90, 0.55), pad=0.0)
    check("tall rectangle: side = height", close(side, 0.80), str(side))
    check("centre of the box stays the centre", close(left + side / 2, 0.50), str(left))


def test_square_crop_verschiebt_statt_zu_schrumpfen() -> None:
    print("\nat the image edge the window shifts instead of shrinking")
    # box in the top left corner; the square would run past the edge
    top, left, side = mc.square_crop((0.00, 0.00, 0.30, 0.50), pad=0.0)
    check("side length is not reduced", close(side, 0.50), str(side))
    check("shifted down, not cut off", close(top, 0.0), str(top))
    check("window lies inside the image", top + side <= 1 + 1e-12)

    # bottom right
    top, left, side = mc.square_crop((0.70, 0.50, 1.00, 1.00), pad=0.0)
    check("bottom right: full side length", close(side, 0.50), str(side))
    check("hineingeschoben", close(top, 0.50) and close(left, 0.50),
          f"top {top}, left {left}")

    # it cannot become larger than the image
    top, left, side = mc.square_crop((0.0, 0.0, 1.0, 1.0), pad=0.5)
    check("side is capped at the image width", close(side, 1.0), str(side))
    check("and then sits at 0.0", close(top, 0.0) and close(left, 0.0))


def test_pad_wirkt_je_seite() -> None:
    print("\npadding is measured per side, then squared off"
          )
    bb = (0.40, 0.20, 0.60, 0.80)          # h = 0.20   w = 0.60
    _, _, side = mc.square_crop(bb, pad=0.10)
    # w grows by 2*0.10*0.60 = 0.12 -> 0.72 ; h grows to 0.24 -> side 0.72
    check("side follows the longer edge after padding", close(side, 0.72),
          str(side))
    _, _, ohne = mc.square_crop(bb, pad=0.0)
    check("padded window larger than unpadded", side > ohne, f"{side:.3f} > {ohne:.3f}")


def test_feste_seitenlaenge() -> None:
    print("\nfixed side length: size constant, position from the mask")
    # Three very differently sized lungs. Only the position may differ.
    bbs = [(0.30, 0.30, 0.50, 0.50),       # small
           (0.05, 0.05, 0.95, 0.95),       # nearly the whole image
           (0.10, 0.60, 0.40, 0.90)]       # small and off to one side
    seiten = [mc.square_crop(bb, pad=0.05, fixed_side=0.80)[2] for bb in bbs]
    check("every window is 0.80 wide", all(close(s, 0.80) for s in seiten),
          str(seiten))
    check("spread of the side length is exactly zero",
          max(seiten) - min(seiten) == 0.0)

    # pad may not change the size any more, otherwise it re-enters through the
    # back door: pad acts on the bounding box, and the bounding box varies.
    a = mc.square_crop(bbs[0], pad=0.00, fixed_side=0.80)
    b = mc.square_crop(bbs[0], pad=0.30, fixed_side=0.80)
    check("pad has no effect on a fixed side", close(a[2], b[2]))

    # The centre still comes from the mask, so the position does vary. That is
    # the residual channel of 0.554 and it is intentional.
    links = [mc.square_crop(bb, pad=0.05, fixed_side=0.80)[1] for bb in bbs]
    check("position still follows the mask", len(set(links)) > 1, str(links))

    # A window that hangs outside is shifted, never shrunk.
    top, left, side = mc.square_crop((0.00, 0.00, 0.10, 0.10), pad=0.0,
                                     fixed_side=0.80)
    check("window at the edge keeps its size", close(side, 0.80), str(side))
    check("and is shifted inwards", close(top, 0.0) and close(left, 0.0))
    check("stays inside the image", top + side <= 1.0 + 1e-12
          and left + side <= 1.0 + 1e-12)

    # Constant downward offset.
    ohne = mc.square_crop((0.30, 0.30, 0.50, 0.50), pad=0.0, fixed_side=0.50)
    mit = mc.square_crop((0.30, 0.30, 0.50, 0.50), pad=0.0, fixed_side=0.50,
                         shift_y=0.03)
    check("shift-y moves the window down by exactly that amount",
          close(mit[0] - ohne[0], 0.03), f"{mit[0]:.4f} vs {ohne[0]:.4f}")
    check("shift-y leaves the size alone", close(mit[2], ohne[2]))
    check("shift-y leaves the horizontal position alone",
          close(mit[1], ohne[1]))

    # The fallback without a mask has to be the same size, not the full image.
    fb = mc.centred_crop(0.80, 0.03)
    check("fallback keeps the fixed size", close(fb[2], 0.80), str(fb[2]))
    check("fallback is centred horizontally", close(fb[1], 0.10), str(fb[1]))
    check("fallback is shifted downwards", close(fb[0], 0.13), str(fb[0]))
    rand = mc.centred_crop(0.98, 0.30)
    check("fallback stays inside the image even with a large offset",
          close(rand[0], 0.02), str(rand[0]))

    # Without the flag nothing changes.
    alt = mc.square_crop((0.40, 0.20, 0.60, 0.80), pad=0.0)
    neu = mc.square_crop((0.40, 0.20, 0.60, 0.80), pad=0.0, fixed_side=None)
    check("default behaviour unchanged", alt == neu, f"{alt} vs {neu}")


def test_transform_boxes() -> None:
    print("\nboxes are carried into the crop")
    S = mc.BOX_SPACE
    # crop: top left quarter of the area
    crop = (0.0, 0.0, 0.5)

    # Box in the middle of that quarter: original x,y = 128, w,h = 128
    # -> inside the crop: 0.25..0.5 of the original = 0.5..1.0 of the crop
    kept, frac = mc.transform_boxes([(0.25 * S, 0.25 * S, 0.25 * S, 0.25 * S)],
                                    crop)
    check("one box remains", len(kept) == 1, str(kept))
    bx, by, bw, bh = kept[0]
    check("x doubles (half the side length)", close(bx, 0.5 * S), str(bx))
    check("width doubles", close(bw, 0.5 * S), str(bw))
    check("fully retained", close(frac, 1.0), str(frac))

    # box completely outside the crop
    kept, frac = mc.transform_boxes([(0.60 * S, 0.60 * S, 0.20 * S, 0.20 * S)],
                                    crop)
    check("box outside the crop is dropped", kept == [], str(kept))
    check("retained fraction 0", close(frac, 0.0), str(frac))

    # box half outside: x from 0.4 to 0.6, the crop ends at 0.5
    kept, frac = mc.transform_boxes([(0.40 * S, 0.10 * S, 0.20 * S, 0.20 * S)],
                                    crop)
    check("box half outside is clipped", len(kept) == 1)
    check("retained fraction is one half", close(frac, 0.5), str(frac))

    # no crop -> nothing changes
    orig = [(100.0, 200.0, 50.0, 60.0)]
    kept, frac = mc.transform_boxes(orig, (0.0, 0.0, 1.0))
    check("without a crop the coordinates are unchanged",
          all(close(a, b, 1e-6) for a, b in zip(kept[0], orig[0])), str(kept))
    check("and the retained fraction is 1", close(frac, 1.0, 1e-9), str(frac))


def test_erhalt_ist_ein_anteil() -> None:
    """The retained fraction has to lie in [0,1]; a unit error pushes it out.

    If the intersection is measured in one coordinate system and the box area
    in another, the ratio leaves the interval. The crop report would then
    state a retention that no box can have, and the mismatch is visible in no
    other number.
    """
    print("\nthe retained fraction really is a fraction")
    S = mc.BOX_SPACE
    rng = np.random.default_rng(0)
    schlimm = 0.0
    for _ in range(300):
        x, y = rng.uniform(0, 0.8, 2) * S
        w, h = rng.uniform(0.05, 0.2, 2) * S
        top, left = rng.uniform(0, 0.5, 2)
        side = rng.uniform(0.3, 1.0 - max(top, left))
        _, frac = mc.transform_boxes([(x, y, w, h)], (top, left, side))
        schlimm = max(schlimm, frac)
        if not (0.0 - 1e-9 <= frac <= 1.0 + 1e-9):
            check("fraction outside [0,1]", False, f"{frac}")
            return
    check("300 random cases stay in [0,1]", True, f"maximum {schlimm:.4f}")


def test_report() -> None:
    print("\nreport")
    rows = [
        {"patientId": "a", "ok": True, "side": 0.5, "box_frac": 1.0},
        {"patientId": "b", "ok": True, "side": 0.8, "box_frac": 0.85},
        {"patientId": "c", "ok": False, "side": 1.0, "box_frac": np.nan},
    ]
    r = mc.crop_report(rows)
    check("n", r["n"] == 3)
    check("empty masks counted", r["n_leer"] == 1)
    check("retention averages only over images with a box", r["n_mit_box"] == 2)
    check("images retaining under 90 % counted", r["n_box_unter_90"] == 1)
    check("zoom is 1/side length", close(r["zoom_median"], 1 / 0.8),
          str(r["zoom_median"]))
    check("empty input gives an empty report", mc.crop_report([]) == {})


def test_konstante_seite_meldet_exakt_null() -> None:
    """The smoke test of the fixed side, pinned so it cannot come back.

    Found on 07.08.2026 while preparing phase 7. The module header promised
    that `side_sd` reads EXACTLY 0.0 under `--fixed-side`. It does not: over
    22872 identical values of 0.800 the standard deviation runs through mean
    and squares and comes out at 1.11e-16. Printed with six decimals that reads
    "0.000000" and looks perfect, while `side_sd == 0.0` is False, so the
    confirming line never appeared. A watchdog keyed to that line would have
    aborted a completely correct run after twenty minutes of cropping.

    The remedy is `side_ptp`, max minus min, which on identical values is
    exactly 0.0 with no arithmetic in between. Both numbers are checked here:
    the new one for being exact, the old one for still being NEAR zero, since
    it stays in the report.
    """
    print("\nconstant side length: the report has to say so")
    rows = [{"patientId": str(i), "ok": True, "side": 0.80,
             "box_frac": np.nan} for i in range(2000)]
    r = mc.crop_report(rows)
    check("side_ptp is exactly zero", r["side_ptp"] == 0.0, repr(r["side_ptp"]))
    check("side_sd is only near zero, which is the whole point",
          abs(r["side_sd"]) < 1e-9, repr(r["side_sd"]))

    import io
    from contextlib import redirect_stdout
    puffer = io.StringIO()
    with redirect_stdout(puffer):
        mc.print_report(r)
    text = puffer.getvalue()
    check("the report confirms the constant window size",
          "CONSTANT WINDOW SIZE" in text, text.strip().splitlines()[-1][:60])

    # And the counterpart: an adaptive window must NOT get that line, otherwise
    # the watchdog would wave through exactly the variant that failed on 26.07.
    rows2 = [{"patientId": str(i), "ok": True, "side": 0.70 + 0.0001 * i,
              "box_frac": np.nan} for i in range(2000)]
    r2 = mc.crop_report(rows2)
    puffer2 = io.StringIO()
    with redirect_stdout(puffer2):
        mc.print_report(r2)
    check("an adaptive window does NOT get that line",
          "CONSTANT WINDOW SIZE" not in puffer2.getvalue(),
          f"side_ptp {r2['side_ptp']:.4f}")


def test_masken_folgen_demselben_fenster() -> None:
    """`rsna_crop_masks.crop_one`, the lung masks in the frame of the crop.

    Two things can go wrong here and neither shows up in a number: the mask
    could be cut with a different window than the image, and it could stop
    being binary. `load_lung` reads it back as `m > 127`, so a smoothing
    interpolation would let the threshold decide where the lung ends instead
    of the segmentation.
    """
    print("\nlung masks follow the same window")
    import rsna_crop_masks as cm

    # A 224 mask with a marked square, cut with a window whose corners are
    # computed by hand: side 0.80, top/left 0.10 -> pixels 22.4 -> 22, side 179.
    m = np.zeros((224, 224), np.uint8)
    m[100:150, 100:150] = 255
    neu = cm.crop_one(m, 0.10, 0.10, 0.80, 224)
    check("output has the requested size", neu.shape == (224, 224),
          str(neu.shape))
    check("the mask stays binary", set(np.unique(neu)) <= {0, 255},
          str(sorted(set(np.unique(neu).tolist()))[:5]))
    # 50 of 179 pixels marked, scaled to 224: about 62.6 pixels on a side.
    n = int((neu == 255).sum())
    erw = (50 / 179 * 224) ** 2
    check("marked area scales with the window",
          abs(n - erw) / erw < 0.05, f"{n} against {erw:.0f}")

    # A window at the edge is pushed inwards, never shrunk. Same rule as
    # square_crop, and it has to be the same rule or masks and images part ways.
    rand = cm.crop_one(m, 0.95, 0.95, 0.80, 224)
    check("window at the edge keeps its size", rand.shape == (224, 224))

    # The full window changes nothing but the resampling.
    ganz = cm.crop_one(m, 0.0, 0.0, 1.0, 224)
    check("the full window returns the mask unchanged",
          bool((ganz == m).all()))

    # Retention is an area ratio in ONE unit. Dividing the two frame fractions
    # would report 1/side^2 on every mask, which is the zoom, not a loss.
    vorher = float((m > 127).mean())
    im_zuschnitt = float((neu > 127).mean())
    erhalt = im_zuschnitt * 0.80 * 0.80 / vorher
    check("retention of an uncut mask is 1.0", abs(erhalt - 1.0) < 0.05,
          f"{erhalt:.4f}")


def test_labels_format() -> None:
    """The rewritten label file has to stay readable by `rsna_train.load_boxes`.

    A renamed or reordered column raises nothing. `load_boxes` then finds no
    boxes at all, and the Grad-CAM evaluation still prints a hit rate.
    """
    print("\nlabel file keeps the original format")
    import tempfile
    from pathlib import Path

    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        src, out = t / "src", t / "out"
        src.mkdir(); out.mkdir()
        pd.DataFrame([
            {"patientId": "a", "x": 100, "y": 100, "width": 50, "height": 50,
             "Target": 1},
            {"patientId": "b", "x": np.nan, "y": np.nan, "width": np.nan,
             "height": np.nan, "Target": 0},
        ]).to_csv(src / "stage_2_train_labels.csv", index=False)

        rows = [
            {"patientId": "a", "ok": True, "side": 1.0, "box_frac": 1.0,
             "_neu": [(100.0, 100.0, 50.0, 50.0)]},
            {"patientId": "b", "ok": True, "side": 1.0, "box_frac": np.nan},
        ]
        path = mc.write_labels(rows, src, out)
        d = pd.read_csv(path)
        check("columns as in the original",
              list(d.columns) == ["patientId", "x", "y", "width", "height",
                                  "Target"], str(list(d.columns)))
        check("negative case is kept as a NaN row",
              bool(d[d.patientId == "b"].x.isna().all()))
        check("Target is carried over",
              int(d[d.patientId == "a"].Target.iloc[0]) == 1)

        b = mc.boxes_by_id(d)
        check("load_boxes logic finds the box again", set(b) == {"a"}, str(b))

        # A positive case whose box the crop removed entirely must not appear
        # as a box row. Grad-CAM would otherwise be scored against nothing.
        rows2 = [{"patientId": "a", "ok": True, "side": 0.2, "box_frac": 0.0}]
        d2 = pd.read_csv(mc.write_labels(rows2, src, out))
        check("box that was cut away becomes a NaN row",
              bool(d2.x.isna().all()), str(d2.values.tolist()))
        check("but stays marked as a positive",
              int(d2.Target.iloc[0]) == 1)


def test_ende_zu_ende() -> None:
    """Real files, a real crop, pixel positions measured by hand.

    The checks above work on the geometry alone. This one writes an image,
    reads it back and locates a marker in it. Swapped axes, or a rounding
    error when [0,1] coordinates are turned into pixels, pass every check
    above and leave an output image that looks unremarkable.
    """
    print("\nend to end: pixels measured")
    import tempfile
    from pathlib import Path

    import pandas as pd
    try:
        from PIL import Image

        import rsna_make_masks as mm
        import cv2                                    # noqa: F401
    except ImportError as e:
        check(f"skipped ({e})", True)
        return

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        img_dir, out, csv = t / "png512", t / "crop512", t / "csv"
        img_dir.mkdir(); csv.mkdir()

        ids, masks = ["a", "b"], []
        for pid in ids:
            im = np.zeros((512, 512), np.uint8)
            im[128:384, 128:384] = 200                # "lung"
            im[200:240, 200:240] = 255                # marker inside it
            Image.fromarray(im).save(img_dir / f"{pid}.png")
            m = np.zeros((256, 256), bool)
            m[64:192, 64:192] = True                  # 0.25 .. 0.75
            masks.append(m)
        mm.save_raw_cache(t / "raw.npz", ids, np.stack(masks))

        pd.DataFrame([
            {"patientId": "a", "x": 400, "y": 400, "width": 100,
             "height": 100, "Target": 1},
            {"patientId": "b", "x": np.nan, "y": np.nan, "width": np.nan,
             "height": np.nan, "Target": 0},
        ]).to_csv(csv / "stage_2_train_labels.csv", index=False)

        rows = mc.run(ids, img_dir, out, t / "raw.npz", "none", 0, 0.05, 512,
                      csv, True)

        r = rows[0]
        # bbox 0.25..0.75 (h = w = 0.5), padding 5 % per side -> 0.55
        check("side length as computed by hand", close(r["side"], 0.55, 1e-9),
              str(r["side"]))
        check("window sits at 0.225", close(r["top"], 0.225, 1e-9)
              and close(r["left"], 0.225, 1e-9), f"{r['top']}, {r['left']}")

        a = np.array(Image.open(out / "a.png"))
        check("output has the requested size", a.shape == (512, 512),
              str(a.shape))
        ys, xs = np.where(a == 255)
        erw0 = (200 / 512 - 0.225) / 0.55 * 512
        erw1 = (240 / 512 - 0.225) / 0.55 * 512
        check("marker lands where predicted (y)",
              abs(ys.min() - erw0) <= 2 and abs(ys.max() - erw1) <= 2,
              f"{ys.min()}..{ys.max()} against {erw0:.0f}..{erw1:.0f}")
        check("and likewise in x (no swapped axes)",
              abs(xs.min() - erw0) <= 2 and abs(xs.max() - erw1) <= 2,
              f"{xs.min()}..{xs.max()}")

        d = pd.read_csv(mc.write_labels(rows, csv, out))
        box = d[d.patientId == "a"].iloc[0]
        # x = 400/1024 = 0.3906 -> (0.3906-0.225)/0.55 = 0.3011 -> 308.4
        check("box moved with the crop", close(box.x, 308.3636, 1e-3), str(box.x))
        check("box width scaled with the crop", close(box.width, 181.8182, 1e-3),
              str(box.width))

        check("both images written",
              sorted(p.name for p in out.glob("*.png")) == ["a.png", "b.png"])


if __name__ == "__main__":
    for fn in (test_mask_bbox, test_square_crop_ist_quadratisch,
               test_square_crop_verschiebt_statt_zu_schrumpfen,
               test_pad_wirkt_je_seite, test_feste_seitenlaenge,
               test_transform_boxes,
               test_erhalt_ist_ein_anteil, test_report,
               test_konstante_seite_meldet_exakt_null,
               test_masken_folgen_demselben_fenster,
               test_labels_format,
               test_ende_zu_ende):
        fn()
    print("\n" + ("-" * 60))
    print("All checks passed." if not FAILED
          else f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
    raise SystemExit(1 if FAILED else 0)
