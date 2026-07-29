"""
Checks the non-Torch logic of rsna_make_masks.py and rsna_cam_lung_check.py.

Why these parts in particular: the finding "peak outside the lung" is built from
three grid conversions (box 1024 -> 224, mask 256 -> 224, heatmap 224) and one
piece of conditional logic. A factor error or a swapped axis does NOT show up in
the finished number. "38 % of the misses lie outside the lung" looks plausible
whether or not the mask sits in the right place. So geometry and assignment are
checked here against hand-built cases whose answer is known in advance.

Both modules import Torch inside the computing functions only. This test
therefore gets by without Torch, deliberately, so that it also runs while a
training job is holding the GPU.

  python test_rsna_masks.py
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import _repo_path  # noqa: F401  (sets sys.path)

import rsna_cam_lung_check as clc
import rsna_make_masks as mm

FAILED: list[str] = []


def check(name: str, cond: bool, info: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FAILED.append(name)


def close(a, b, tol=1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------

def test_box_geometry() -> None:
    print("\nBox geometry 1024 -> 224")

    # Whole image: must give the full area.
    m = clc.box_mask([(0, 0, 1024, 1024)], 224, 1024)
    check("full box covers everything", m.all() and close(m.mean(), 1.0))

    # Upper left quarter in the original grid -> upper left quarter at 224.
    m = clc.box_mask([(0, 0, 512, 512)], 224, 1024)
    check("quarter box -> area 0.25", close(m.mean(), 0.25),
          f"measured {m.mean():.4f}")
    check("quarter box sits at the upper left",
          m[0, 0] and m[111, 111] and not m[112, 112] and not m[223, 223])

    # Axes not swapped: narrow in x, tall in y.
    m = clc.box_mask([(0, 0, 1024, 512)], 224, 1024)   # full width, half height
    check("x/y not swapped", m[0, :].all() and not m[223, :].any(),
          "full width at the top, empty at the bottom")

    # Negative coordinates are clamped instead of wrapping around (Python slicing
    # with a negative start would otherwise be a silent error at the image edge).
    m = clc.box_mask([(-100, -100, 300, 300)], 224, 1024)
    check("negative origin is clamped", m[0, 0] and m.sum() > 0)

    # Two boxes are unioned, overlapping area is not counted twice.
    m2 = clc.box_mask([(0, 0, 512, 512), (0, 0, 512, 512)], 224, 1024)
    check("duplicated box counts once", close(m2.mean(), 0.25))


def test_analyse_one() -> None:
    print("\nanalyse_one: assignment of the peak to box / lung")
    size = 20
    box = np.zeros((size, size), bool); box[2:6, 2:6] = True
    lung = np.zeros((size, size), bool); lung[0:10, 0:10] = True

    # Case A: peak inside the box and inside the lung.
    heat = np.zeros((size, size)); heat[3, 3] = 1.0
    r = clc.analyse_one(heat, box, lung)
    check("A peak in box", r["peak_in_box"])
    check("A peak in lung", r["peak_in_lung"])
    check("A mass in box = 1", close(r["mass_in_box"], 1.0))

    # Case B: peak outside the lung (image border), second best point inside the
    # box. Exactly the case in which a crop would buy something.
    heat = np.zeros((size, size)); heat[19, 19] = 1.0; heat[3, 3] = 0.9
    r = clc.analyse_one(heat, box, lung)
    check("B peak not in box", not r["peak_in_box"])
    check("B peak not in lung", not r["peak_in_lung"])
    check("B lung-restricted peak hits the box", r["peak_in_box_lungrestricted"],
          "-> headroom for a crop")

    # Case C: peak inside the lung but beside the box. A crop changes nothing.
    heat = np.zeros((size, size)); heat[8, 8] = 1.0; heat[3, 3] = 0.5
    r = clc.analyse_one(heat, box, lung)
    check("C peak not in box", not r["peak_in_box"])
    check("C peak inside the lung", r["peak_in_lung"])
    check("C lung-restricted peak still misses",
          not r["peak_in_box_lungrestricted"], "-> no headroom")

    # Areas and box_in_lung: the box lies fully inside the lung -> 1.0.
    check("box_in_lung = 1 at full coverage", close(r["box_in_lung"], 1.0))
    check("lung_area correct", close(r["lung_area"], 100 / 400))
    check("box_area correct", close(r["box_area"], 16 / 400))

    # Box covered by half -> 0.5. That is the control number against
    # undersegmentation; it has to be right.
    lung_half = np.zeros((size, size), bool); lung_half[0:4, :] = True
    r = clc.analyse_one(heat, box, lung_half)
    check("box_in_lung = 0.5 at half coverage",
          close(r["box_in_lung"], 0.5), f"measured {r['box_in_lung']:.3f}")

    # Empty heatmap -> None instead of a division by zero.
    check("empty heatmap gives None",
          clc.analyse_one(np.zeros((size, size)), box, lung) is None)

    # Negative CAM values are clamped, not counted as mass.
    heat = np.zeros((size, size)); heat[3, 3] = 1.0; heat[15, 15] = -5.0
    r = clc.analyse_one(heat, box, lung)
    check("negative values are clamped", close(r["mass_in_box"], 1.0))

    # Empty lung mask: no restriction, no crash.
    r = clc.analyse_one(heat, box, np.zeros((size, size), bool))
    check("empty mask falls back to the unrestricted peak",
          r["peak_in_box_lungrestricted"] == r["peak_in_box"])
    check("empty mask -> lung_area 0", close(r["lung_area"], 0.0))


def test_summarise() -> None:
    print("\nsummarise: breakdown of the misses")
    def row(hit, in_lung, restr, mass_lung):
        return dict(peak_in_box=hit, peak_in_lung=in_lung,
                    peak_in_box_lungrestricted=restr,
                    mass_in_box=0.5 if hit else 0.1, mass_in_lung=mass_lung,
                    box_area=0.2, lung_area=0.4, box_in_lung=1.0,
                    null_free=0.2, null_restricted=0.5)

    # 2 hits, 3 misses, 2 of them outside the lung.
    df = pd.DataFrame([
        row(True,  True,  True,  0.8),
        row(True,  True,  True,  0.8),
        row(False, False, True,  0.3),
        row(False, False, False, 0.3),
        row(False, True,  False, 0.9),
    ])
    s = clc.summarise(df)
    check("n", s["n"] == 5)
    check("hit rate 0.4", close(s["peak_in_box"], 0.4))
    check("misses counted", s["n_miss"] == 3)
    check("misses outside = 2/3", close(s["miss_outside_lung"], 2 / 3),
          f"measured {s['miss_outside_lung']:.4f}")
    check("lung lift = 0.6 - 0.4", close(s["peak_in_lung_lift"], 0.2))
    check("headroom = 0.6 - 0.4", close(s["crop_headroom"], 0.2),
          "lung-restricted 3/5 against 2/5")

    # With no misses nothing may crash; NaN is the expected answer.
    s2 = clc.summarise(df[df.peak_in_box].reset_index(drop=True))
    check("no misses -> NaN instead of a crash", np.isnan(s2["miss_outside_lung"]))


def test_cv_mean() -> None:
    print("\ncv_mean: mean across folds")
    rows = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    m, s = clc.cv_mean(rows, "x")
    check("Mittel", close(m, 2.0))
    check("SD with ddof=1", close(s, 1.0))
    m, s = clc.cv_mean([{"x": float("nan")}, {"x": 4.0}], "x")
    check("NaN is passed over", close(m, 4.0))
    m, _ = clc.cv_mean([{"x": float("nan")}], "x")
    check("only NaN -> NaN", np.isnan(m))


def test_ids_from_csvs() -> None:
    print("\nrsna_make_masks: ID selection")
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        pd.DataFrame({"patientId": ["b", "a", "a"]}).to_csv(t / "cam_f0_s0.csv",
                                                           index=False)
        pd.DataFrame({"patientId": ["c", "a"]}).to_csv(t / "cam_f1_s0.csv",
                                                       index=False)
        ids = mm.ids_from_csvs([str(t / "cam_f*_s0.csv")])
        check("glob catches both folds, deduplicated, sorted",
              ids == ["a", "b", "c"], str(ids))

        try:
            mm.ids_from_csvs([str(t / "gibtsnicht_*.csv")])
            check("empty glob raises", False)
        except FileNotFoundError:
            check("empty glob raises FileNotFoundError", True)

        pd.DataFrame({"foo": [1]}).to_csv(t / "bad.csv", index=False)
        try:
            mm.ids_from_csvs([str(t / "bad.csv")])
            check("missing column raises", False)
        except ValueError:
            check("missing column raises ValueError", True)


def test_pending_jobs() -> None:
    print("\nrsna_make_masks: what is still to be computed")
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "src", Path(tmp) / "dst"
        src.mkdir(); dst.mkdir()
        for pid in ("a", "b", "c"):
            (src / f"{pid}.png").write_bytes(b"x")
        (dst / "a.png").write_bytes(b"x")            # present already

        jobs, skipped, missing = mm.pending_jobs(["a", "b", "c", "weg"], src, dst,
                                                 overwrite=False)
        check("existing mask is skipped", skipped == 1)
        check("missing source image is counted", missing == 1)
        check("two images to compute", len(jobs) == 2,
              str([j[0].stem for j in jobs]))

        jobs, skipped, _ = mm.pending_jobs(["a", "b", "c"], src, dst, overwrite=True)
        check("--overwrite recomputes everything", len(jobs) == 3 and skipped == 0)


def test_area_report() -> None:
    print("\nrsna_make_masks: area statistics")
    a = np.array([0.01, 0.30, 0.35, 0.40, 0.90])
    r = mm.area_report(a)
    check("n", r["n"] == 5)
    check("Median", close(r["median"], 0.35))
    check("empty mask detected (<0.05)", r["n_empty"] == 1)
    check("huge mask detected (>0.60)", r["n_huge"] == 1)
    check("thresholds are strict: 0.05 does not count as empty",
          mm.area_report(np.array([0.05, 0.05]))["n_empty"] == 0)
    check("empty array -> empty dict", mm.area_report(np.array([])) == {})


def test_packing() -> None:
    print("\nRaw cache: bit packing")
    rng = np.random.default_rng(0)
    m = rng.random((7, 256, 256)) > 0.5
    back = mm.unpack_masks(mm.pack_masks(m))
    check("round trip is lossless", bool((back == m).all()))
    check("factor of 8 smaller", mm.pack_masks(m).nbytes * 8 == m.nbytes,
          f"{mm.pack_masks(m).nbytes} vs {m.nbytes} bytes")
    # A bit offset would be the classic silent error: the mask still looks
    # plausible but is shifted by one row.
    one = np.zeros((1, 256, 256), bool); one[0, 3, 5] = True
    b = mm.unpack_masks(mm.pack_masks(one))[0]
    check("position is preserved exactly",
          b[3, 5] and b.sum() == 1, f"sum {b.sum()}")


def test_refine_variant() -> None:
    print("\nRefinement variants")
    raw = np.zeros((256, 256), bool)
    raw[60:200, 40:100] = True          # left lung
    raw[60:200, 156:216] = True         # right lung

    a_none = mm.refine_variant(raw, "none", 0).mean()
    a_def = mm.refine_variant(raw, "default", 0).mean()
    a_hull = mm.refine_variant(raw, "hull", 0).mean()
    a_dil = mm.refine_variant(raw, "default", 6).mean()
    check("dilation enlarges the area", a_dil > a_def,
          f"{a_def:.4f} -> {a_dil:.4f}")
    check("hull is never smaller than default", a_hull >= a_def - 1e-9,
          f"{a_def:.4f} vs {a_hull:.4f}")
    check("'none' leaves the raw mask unchanged", close(a_none, raw.mean()))
    check("dilation is monotone",
          mm.refine_variant(raw, "default", 2).mean()
          <= mm.refine_variant(raw, "default", 8).mean())

    # The consolidation case in two forms, and the two behave differently. That
    # is not hair-splitting; it decides whether 'hull' is any use as a remedy:
    #
    #   a) An ENCLOSED hole in the middle of the lung. `_clean` already fills
    #      that via binary_fill_holes, so 'hull' changes nothing.
    #   b) A notch OPEN TO THE BORDER (the consolidation reaches the pleura or
    #      the diaphragm). fill_holes cannot do that, because the gap is
    #      connected to the background. Only the convex hull brings it back.
    #      This is the shape a real lower lobe pneumonia has.
    enclosed = raw.copy(); enclosed[100:150, 50:90] = False
    check("enclosed hole: 'default' already fills it",
          close(mm.refine_variant(enclosed, "default", 0).mean(), a_def),
          "-> 'hull' is not needed for that")

    notched = raw.copy(); notched[100:150, 40:90] = False      # reaches the border
    n_def = mm.refine_variant(notched, "default", 0).mean()
    n_hull = mm.refine_variant(notched, "hull", 0).mean()
    check("border-open notch: 'default' does not bring it back",
          n_def < a_def - 1e-6, f"{n_def:.4f} < {a_def:.4f}")
    check("border-open notch: 'hull' brings area back", n_hull > n_def,
          f"{n_def:.4f} -> {n_hull:.4f}")

    try:
        mm.refine_variant(raw, "quatsch", 0)
        check("unknown mode raises", False)
    except ValueError:
        check("unknown mode raises ValueError", True)

    out = mm.to_out(mm.refine_variant(raw, "default", 0))
    check("to_out returns 224x224", out.shape == (224, 224), str(out.shape))
    check("to_out stays binary", set(np.unique(out)) <= {0, 255},
          str(np.unique(out)))


def test_null_baselines() -> None:
    print("\nNull baselines (the error of the first version)")
    size = 20
    box = np.zeros((size, size), bool); box[2:6, 2:6] = True      # area 0.04
    lung = np.zeros((size, size), bool); lung[0:10, 0:10] = True  # area 0.25
    heat = np.zeros((size, size)); heat[3, 3] = 1.0
    r = clc.analyse_one(heat, box, lung)

    check("null_free = box area", close(r["null_free"], 0.04))
    # Box lies entirely inside the lung -> chance hit in the lung = 0.04/0.25
    check("null_restricted = (box AND lung)/lung",
          close(r["null_restricted"], 0.04 / 0.25),
          f"measured {r['null_restricted']:.4f}")
    check("peak coordinates are stored",
          r["peak_y"] == 3 and r["peak_x"] == 3)

    # A fold in which the outside share matches chance EXACTLY: the lift has to
    # be 0, rather than the raw value looking large.
    n = 100
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n):
        rows.append(dict(peak_in_box=False, peak_in_lung=bool(rng.random() < 0.25),
                         peak_in_box_lungrestricted=False, mass_in_box=0.1,
                         mass_in_lung=0.25, box_area=0.04, lung_area=0.25,
                         box_in_lung=1.0, null_free=0.04, null_restricted=0.16))
    s = clc.summarise(pd.DataFrame(rows))
    check("raw outside value is large (~0.75)", s["miss_outside_lung"] > 0.6,
          f"{s['miss_outside_lung']:.3f}")
    check("lift against the null is ~0", abs(s["miss_outside_lift"]) < 0.12,
          f"{s['miss_outside_lift']:+.3f}  <- exactly the number that was missing")
    check("headroom_null is computed", close(s["headroom_null"], 0.12),
          f"{s['headroom_null']:.3f}")


def test_paired_t() -> None:
    print("\npaired t across folds")
    rows = [{"d": 0.08}, {"d": 0.07}, {"d": 0.09}, {"d": 0.08}, {"d": 0.08}]
    check("consistent difference -> large t", clc.paired_t(rows, "d") > 2.78,
          f"t={clc.paired_t(rows, 'd'):.2f}")
    rows = [{"d": -0.12}, {"d": -0.06}, {"d": 0.05}, {"d": 0.12}, {"d": 0.08}]
    check("scattered difference -> small t", abs(clc.paired_t(rows, "d")) < 2.78,
          f"t={clc.paired_t(rows, 'd'):.2f}   (the real case)")
    check("a single value -> NaN", np.isnan(clc.paired_t([{"d": 1.0}], "d")))


def test_parse_variant() -> None:
    print("\nSweep: variant parser")
    import rsna_mask_sweep as sw
    check("'hull:4'", sw.parse_variant("hull:4") == ("hull", 4))
    check("without a colon -> 0 pixels", sw.parse_variant("default") == ("default", 0))
    check("'none:0'", sw.parse_variant("none:0") == ("none", 0))


def test_balanced_sample() -> None:
    print("\nrsna_make_masks: balanced sample")
    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "splits.json"
        labels = {f"p{i}": (1 if i < 20 else 0) for i in range(100)}
        holdout = ["p0", "p1", "p90", "p91"]
        sp.write_text(json.dumps({"labels": labels, "holdout": holdout}))

        got = mm.balanced_sample(sp, 5, seed=0)
        check("2x5 images", len(got) == 10, str(len(got)))
        pos = sum(labels[g] for g in got)
        check("balanced 5/5", pos == 5, f"{pos} positive")
        check("holdout excluded", not (set(got) & set(holdout)),
              "otherwise the reserved set would be touched silently")
        check("deterministic for the same seed",
              mm.balanced_sample(sp, 5, seed=0) == got)
        check("different seed -> different selection",
              mm.balanced_sample(sp, 5, seed=1) != got)

        # More requested than available: take what is there instead of raising.
        many = mm.balanced_sample(sp, 1000, seed=0)
        check("over-request is capped", len(many) == 96,
              f"{len(many)} of 100 minus 4 holdout")


def test_load_boxes_matches_train() -> None:
    """The sweep keeps its own copy of load_boxes so that it runs without Torch.
    This check stops the two from drifting apart: a diverged copy would silently
    score a different set of boxes."""
    print("\nSweep: load_boxes copy")
    import rsna_mask_sweep as sw
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        pd.DataFrame({
            "patientId": ["a", "a", "b", "c", "d"],
            "x": [10, 50, 20, np.nan, 5],
            "y": [10, 50, 20, 30, 5],
            "width": [100, 100, 200, 40, 10],
            "height": [100, 100, 200, 40, 10],
            "Target": [1, 1, 1, 1, 0],
        }).to_csv(t / "stage_2_train_labels.csv", index=False)
        got = sw.load_boxes(t)
        check("only Target==1 with valid coordinates",
              set(got) == {"a", "b"}, str(sorted(got)))
        check("two boxes for 'a'", len(got["a"]) == 2)
        check("row with a NaN coordinate is dropped", "c" not in got)
        check("negative case (Target=0) is dropped", "d" not in got)
        check("coordinates as float", isinstance(got["b"][0][0], float))

        try:
            from rsna_train import load_boxes as orig
        except Exception as e:                       # no Torch -> not checkable
            print(f"  --    comparison with rsna_train skipped ({type(e).__name__})")
            return
        check("identical to rsna_train.load_boxes", orig(t) == got)
        from rsna_train import BOX_SPACE as ORIG_SPACE
        check("BOX_SPACE identical", sw.BOX_SPACE == ORIG_SPACE)


def test_grouped_bootstrap() -> None:
    """The grouped bootstrap has to give a WIDER interval than the per-image one.
    Otherwise the grouping does not take hold, and the external number would look
    more certain than it is."""
    print("\nExternal: grouped bootstrap")
    import rsna_external_kermany as ext

    rng = np.random.default_rng(0)
    n_groups, per_group = 40, 10
    groups, y, p = [], [], []
    for g in range(n_groups):
        cls = g % 2
        # Nearly identical scores within one group: that is what several films
        # of the same child look like.
        base = rng.normal(0.6 if cls else 0.4, 0.25)
        for _ in range(per_group):
            groups.append(f"g{g}"); y.append(cls)
            p.append(base + rng.normal(0, 0.01))
    y, p, groups = np.array(y), np.array(p), np.array(groups)

    a_g, lo_g, hi_g = ext.grouped_bootstrap_auc(y, p, groups, B=200, seed=0)
    a_i, lo_i, hi_i = ext.grouped_bootstrap_auc(
        y, p, np.arange(len(y)).astype(str), B=200, seed=0)   # each row its own group
    check("point estimate identical", close(a_g, a_i, 1e-9))
    check("grouped interval is WIDER", (hi_g - lo_g) > (hi_i - lo_i),
          f"grouped {hi_g - lo_g:.3f} vs per image {hi_i - lo_i:.3f}")


def test_pad_to_square() -> None:
    print("\nExternal: PadToSquare")
    from PIL import Image as PILImage

    import rsna_external_kermany as ext

    a = np.zeros((60, 100), np.uint8); a[:] = 200
    img = PILImage.fromarray(a, mode="L")
    out = ext.PadToSquare()(img)
    check("becomes square", out.size == (100, 100), str(out.size))
    o = np.asarray(out)
    check("image content kept and centred", bool((o[20:80, :] == 200).all()))
    check("padding is the median, not black", int(o[0, 0]) == 200,
          f"corner {o[0, 0]}, black would itself be a feature")
    sq = PILImage.fromarray(np.zeros((50, 50), np.uint8), mode="L")
    check("square image stays unchanged",
          ext.PadToSquare()(sq).size == (50, 50))


def test_stratified_by_score() -> None:
    print("\nExternal: stratification by the leak score")
    import rsna_external_kermany as ext

    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n)

    # IMPORTANT for the construction of this test: the noise has to be LARGE
    # against the class separation. With small noise the score falls apart into
    # two separate peaks, each quintile then holds one class only, and all that
    # is measured is the edge quintile, so the stratification runs empty. That is
    # exactly what the first version of this test failed on.
    leak = 1.0 * y + rng.normal(0, 1.0, n)

    raw = roc_auc_score(y, leak)
    a, per = ext.stratified_by_score(y, leak, leak)
    check("report holds n_pos, n_neg and discordant pairs per quintile",
          len(per[0]) == 6, str(per[0]))
    check("a pure leak score collapses under stratification", a < raw - 0.15,
          f"raw {raw:.3f} -> stratified {a:.3f}")
    check("and lands close to chance", abs(a - 0.5) < 0.10, f"{a:.3f}")

    # Score with real added signal: it holds up inside the quintiles too.
    good = leak + 2.0 * y
    raw_g = roc_auc_score(y, good)
    a2, _ = ext.stratified_by_score(y, good, leak)
    check("real added signal survives the stratification", a2 > 0.70,
          f"raw {raw_g:.3f} -> stratified {a2:.3f}")
    check("and lies clearly above the pure leak", a2 > a + 0.15,
          f"{a2:.3f} against {a:.3f}")

    # The error from the first real run: an almost single-class quintile (1182
    # positive, 1 negative) entered the mean with its full n weight. Weighted by
    # discordant pairs it drops out for practical purposes.
    n2 = 2000
    y2 = np.concatenate([rng.integers(0, 2, n2), np.ones(n2, int)])
    strat2 = np.concatenate([rng.normal(0, 1, n2), rng.normal(9, 0.1, n2)])
    y2[n2 + 5] = 0                                   # exactly ONE negative up top
    good = np.where(y2 == 1, 3.0, 0.0) + rng.normal(0, 1, len(y2))
    junk = rng.normal(0, 1, len(y2))                 # pure noise up top
    p2 = np.where(strat2 > 5, junk, good)
    a3, per3 = ext.stratified_by_score(y2, p2, strat2)
    tail = [r for r in per3 if r[3] < 30]            # quintiles with <30 negatives
    check("almost single-class quintile is detected", len(tail) >= 1,
          f"{len(tail)} quintile(s) with fewer than 30 negatives")
    n_weighted = float(np.average([r[5] for r in per3 if not np.isnan(r[5])],
                                  weights=[r[1] for r in per3
                                           if not np.isnan(r[5])]))
    check("pair weighting ignores the noise quintile, n weighting does not",
          abs(a3 - 0.5) > abs(n_weighted - 0.5) - 1e-9,
          f"pairs {a3:.3f} vs n weighted {n_weighted:.3f}")


def test_operating_point() -> None:
    print("\nExternal: operating point")
    import rsna_external_kermany as ext

    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([.9, .8, .2, .7, .1, .1, .1, .1])
    op = ext.operating_point(y, p, 0.5)
    check("sensitivity 2/3", close(op["sens"], 2 / 3))
    check("specificity 4/5", close(op["spec"], 4 / 5))
    check("PPV 2/3", close(op["ppv"], 2 / 3))
    check("positive rate 3/8", close(op["pos_rate"], 3 / 8))


def test_no_device_map_location() -> None:
    """No RSNA script may load a checkpoint straight onto the device.

    This is a source-level check and not a behavioural one, deliberately: the bug
    appears ONLY on real DirectML hardware, so it cannot be triggered in a CPU
    environment. A behavioural test would pass on Linux and prove nothing. The
    bug cost one aborted run and announces itself as

        TypeError: '>=' not supported between instances of 'torch.device' and 'int'

    which looks like a broken checkpoint. Torch passes `map_location` on to
    `torch_directml.device()`, which expects an integer there. The right way is
    always: load onto the CPU, model onto the device, let it copy.
    """
    print("\nDirectML: never load a checkpoint straight onto the device")
    import ast

    # Via the AST rather than the raw text: otherwise the check trips on the
    # docstrings that EXPLAIN the bug. What counts is real calls carrying the
    # map_location keyword, nothing else.
    bad, checked = [], 0
    for f in sorted(Path(".").glob("rsna_*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "map_location":
                    continue
                checked += 1
                ok = isinstance(kw.value, ast.Constant) and kw.value.value == "cpu"
                if not ok:
                    bad.append(f"{f.name}:{kw.value.lineno}")
    check("all rsna_*.py load onto 'cpu'", not bad,
          "; ".join(bad) if bad else f"{checked} calls checked")


def test_cache_resume() -> None:
    """The resume path of the big mask run.

    The failure this guards against: an aborted run wrote the mask PNG but never
    saved the raw cache afterwards. If the next run looks at the PNG only, the
    image counts as done and is missing from the cache for ever. The crop step
    notices that hours later, and then as a missing image rather than as an
    error.
    """
    print("\nrsna_make_masks: raw cache and resume")
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        src, dst = t / "src", t / "dst"
        src.mkdir(); dst.mkdir()
        for pid in ("a", "b", "c"):
            (src / f"{pid}.png").write_bytes(b"x")
            (dst / f"{pid}.png").write_bytes(b"x")   # all three "computed"

        cache = t / "raw.npz"
        rng = np.random.default_rng(0)
        masks = rng.random((2, mm.SEG_SIZE, mm.SEG_SIZE)) > 0.5
        mm.save_raw_cache(cache, ["a", "b"], masks)   # but only two saved

        check("cached_ids reads the cache", mm.cached_ids(cache) == {"a", "b"})
        check("cached_ids without a file -> empty", mm.cached_ids(t / "weg.npz") == set())
        check("cached_ids without a path -> empty", mm.cached_ids(None) == set())

        jobs, skipped, _ = mm.pending_jobs(["a", "b", "c"], src, dst,
                                           overwrite=False,
                                           cached=mm.cached_ids(cache))
        check("only the unsaved image is recomputed",
              [j[0].stem for j in jobs] == ["c"], str([j[0].stem for j in jobs]))
        check("the saved ones count as done", skipped == 2)

        jobs, skipped, _ = mm.pending_jobs(["a", "b", "c"], src, dst,
                                           overwrite=False, cached=None)
        check("without cache bookkeeping the PNG alone decides",
              len(jobs) == 0 and skipped == 3)

        # Adding on: 'c' comes in, 'a' and 'b' must not get lost.
        mm.save_raw_cache(cache, ["c"],
                          rng.random((1, mm.SEG_SIZE, mm.SEG_SIZE)) > 0.5)
        ids2, packed2 = mm.load_raw_cache(cache)
        check("adding on keeps the old entries", ids2 == ["a", "b", "c"],
              str(ids2))
        check("content of 'a' unchanged",
              bool((mm.unpack_masks(packed2[0:1])[0] == masks[0]).all()))
        check("no .tmp.npz is left lying around", list(t.glob("*.tmp.npz")) == [])


def test_npz_handles_closed() -> None:
    """Every np.load on a .npz has to sit inside a `with`.

    What set this off: for a .npz `np.load` returns a LAZY `NpzFile` and holds
    the file open. On Windows `os.replace` on an open file fails with "WinError
    5: access denied", in the middle of the big mask run, at the first
    intermediate save.

    Why a source-level check and not a behavioural test: on Linux the faulty code
    runs through without complaint. A test that is green here and red on the
    target machine is no test. Checked are all files that call `os.replace`
    themselves, since only there is an open handle fatal.
    """
    print("\nWindows: .npz handles have to be closed")
    root = Path(mm.__file__).parent
    betroffen = [f for f in sorted(root.glob("rsna_*.py"))
                 if "os.replace" in f.read_text(encoding="utf-8", errors="replace")]
    check("at least one file replaces files", bool(betroffen),
          ", ".join(f.name for f in betroffen))

    bad = []
    for f in betroffen:
        tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        drin = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    drin.add(id(item.context_expr))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr == "load"
                    and isinstance(fn.value, ast.Name) and fn.value.id == "np"):
                continue
            if id(node) not in drin:
                bad.append(f"{f.name}:{node.lineno}")
    check("no np.load outside a with", not bad,
          "; ".join(bad) if bad else "all closed")


def test_resolve_device() -> None:
    """Device resolution: the bug that stopped the big mask run.

    `--device directml` was handed to `.to()` raw. But "directml" is not a Torch
    device name; training translates it through `pick_device`, this script did
    not.

    The second part is the more dangerous one: the resolution has to be
    idempotent. A second call on an already resolved device otherwise falls
    through every name comparison and silently returns the CPU. The run would go
    through, twenty times slower, with no error message.
    """
    print("\nrsna_make_masks: device resolution")
    try:
        import torch
    except ImportError:
        check("  -- skipped (no Torch)", True)
        return

    cpu = mm.resolve_device("cpu")
    check("'cpu' becomes a Torch device", isinstance(cpu, torch.device),
          str(type(cpu)))
    check("and specifically the CPU", cpu.type == "cpu", str(cpu))

    again = mm.resolve_device(cpu)
    check("idempotent: a resolved device comes back unchanged", again is cpu,
          str(again))

    class FremdesGeraet:                       # stands in for torch_directml.device()
        pass
    fremd = FremdesGeraet()
    check("a non-Torch device is passed through too",
          mm.resolve_device(fremd) is fremd)

    src = Path(mm.__file__).read_text(encoding="utf-8", errors="replace")
    check("no raw device string reaching .to()",
          ".to(device)" not in src,
          "still present" if ".to(device)" in src else "keiner")


def main() -> int:
    for fn in (test_box_geometry, test_analyse_one, test_summarise, test_cv_mean,
               test_ids_from_csvs, test_pending_jobs, test_cache_resume,
               test_npz_handles_closed, test_resolve_device, test_area_report,
               test_packing, test_refine_variant, test_null_baselines,
               test_paired_t, test_parse_variant, test_balanced_sample,
               test_load_boxes_matches_train, test_grouped_bootstrap,
               test_pad_to_square, test_stratified_by_score,
               test_operating_point, test_no_device_map_location):
        fn()
    print("\n" + ("-" * 60))
    if FAILED:
        print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
