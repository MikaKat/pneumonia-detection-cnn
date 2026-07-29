"""
Checks the geometry of `rsna_mask_silhouette.py` on masks built by hand.

The headline number of that script is that the cutout between the lungs
reveals the projection at AUC 0.69, where 0.5 would mean the cutout says
nothing about it. The claim holds only if the cutout is taken row by row.
Defined over one global bounding box instead, it would take in areas above
the lung apices and below the costophrenic angles that are not mediastinum,
which flatters the result. A number obtained that way reads as plausibly as
the right one.

  python test_rsna_silhouette.py
"""

from __future__ import annotations

import numpy as np

import _repo_path  # noqa: F401  (sets sys.path)

import rsna_mask_silhouette as ms

FAILED: list[str] = []


def check(name: str, cond: bool, info: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FAILED.append(name)


def zwei_lungen(H=100, W=100, oben=20, unten=80, links=(10, 40),
                rechts=(60, 90)) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[oben:unten, links[0]:links[1]] = True
    m[oben:unten, rechts[0]:rechts[1]] = True
    return m


def test_inner_gap() -> None:
    print("\ncutout between the lungs")
    m = zwei_lungen()
    g = ms.inner_gap(m)
    check("cutout sits between the lungs",
          bool(g[50, 40:60].all()), str(g[50].sum()))
    check("width is correct (columns 40..59)", int(g[50].sum()) == 20,
          str(int(g[50].sum())))
    check("lung itself is excluded", not bool((g & m).any()))
    check("nothing above the lung apices", not bool(g[:20].any()),
          "a global bounding box would include this")
    check("nothing below the costophrenic angles", not bool(g[80:].any()))
    check("nothing lateral to either lung",
          not bool(g[:, :10].any()) and not bool(g[:, 90:].any()))

    check("empty mask -> empty cutout",
          not ms.inner_gap(np.zeros((10, 10), bool)).any())

    einzeln = np.zeros((10, 10), bool)
    einzeln[5, 3:7] = True
    check("one connected region has no cutout",
          not ms.inner_gap(einzeln).any())


def test_silhouette_features() -> None:
    print("\nmeasures taken on the cutout")
    m = zwei_lungen()                       # lungs 30 wide each, gap 20
    f = ms.silhouette_features(m)
    check("ctr = gap / total width = 20/80", abs(f["ctr"] - 0.25) < 1e-9,
          str(f["ctr"]))
    # cutout: rows 20..79 (60) x columns 40..59 (20) = 1200 of 10000
    check("mid_area = 0.12", abs(f["mid_area"] - 0.12) < 1e-9, str(f["mid_area"]))
    check("mid_top = 0 (cutout starts at the apices)",
          abs(f["mid_top"]) < 1e-9, str(f["mid_top"]))

    # Wider mediastinum -> larger ctr
    breit = zwei_lungen(links=(10, 30), rechts=(70, 90))
    check("wider cutout -> larger ctr",
          ms.silhouette_features(breit)["ctr"] > f["ctr"],
          f"{ms.silhouette_features(breit)['ctr']:.3f} > {f['ctr']:.3f}")

    check("mask too small -> None",
          ms.silhouette_features(np.zeros((100, 100), bool)) is None)


def test_mid_top() -> None:
    """A cutout that starts further down has to give a larger mid_top.

    The denominator is `y1 - y0`, the DISTANCE between the outermost lung
    rows (here 79 - 20 = 59), not their count (60). Either convention would
    be defensible. This test pins down the one in use, because the two differ
    by less than one per cent, and a recomputation that assumes the other
    reads that gap as an error.
    """
    print("\nvertical position of the cutout")
    m = np.zeros((100, 100), bool)
    m[20:80, 10:40] = True
    m[20:80, 60:90] = True
    m[20:40, 40:60] = True            # joined at the top -> cutout from row 40
    f = ms.silhouette_features(m)
    check("mid_top = (40-20)/(79-20)", abs(f["mid_top"] - 20 / 59) < 1e-9,
          str(f["mid_top"]))

    tiefer = m.copy()
    tiefer[40:60, 40:60] = True       # cutout only begins at row 60
    check("cutout starting lower -> larger mid_top",
          ms.silhouette_features(tiefer)["mid_top"] > f["mid_top"],
          f"{ms.silhouette_features(tiefer)['mid_top']:.3f} > {f['mid_top']:.3f}")


def test_box_location() -> None:
    print("\nwhere the box lies")
    m = zwei_lungen()                 # 100x100; lung x 10..39 and 60..89
    S = ms.BOX_SPACE
    # Box completely inside the left lung: x 15..35, y 30..50
    r = ms.box_location(m, [(0.15 * S, 0.30 * S, 0.20 * S, 0.20 * S)])
    check("entirely inside the lung", abs(r["in_lunge"] - 1.0) < 1e-9, str(r))
    # Box completely inside the cutout: x 42..58
    r = ms.box_location(m, [(0.42 * S, 0.30 * S, 0.16 * S, 0.20 * S)])
    check("entirely inside the cutout", abs(r["in_mitte"] - 1.0) < 1e-9, str(r))
    # Box far out on the left (x 0..8) -> neither of the two
    r = ms.box_location(m, [(0.0, 0.30 * S, 0.08 * S, 0.20 * S)])
    check("outside both", abs(r["ausserhalb"] - 1.0) < 1e-9, str(r))
    # Box spanning all three regions: the fractions still sum to 1
    r = ms.box_location(m, [(0.05 * S, 0.25 * S, 0.60 * S, 0.40 * S)])
    check("fractions add up to 1",
          abs(r["in_lunge"] + r["in_mitte"] + r["ausserhalb"] - 1.0) < 1e-9,
          str(r))
    check("without a box -> None", ms.box_location(m, []) is None)


def test_auc() -> None:
    # The AUC has to match a reference implementation to the last digit.
    # Otherwise the 0.69 in the report and the 0.5 it is compared against
    # come from two different definitions.
    print("\nAUC against sklearn")
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        check("skipped (no sklearn)", True)
        return
    rng = np.random.default_rng(0)
    for n in (60, 400):
        y = (rng.random(n) > 0.6).astype(int)
        s = rng.random(n)
        check(f"n={n}", abs(ms.auc(y, s) - roc_auc_score(y, s)) < 1e-12,
              f"{ms.auc(y, s):.10f}")
    check("only one class -> nan", np.isnan(ms.auc(np.zeros(5, int),
                                                   rng.random(5))))


if __name__ == "__main__":
    for fn in (test_inner_gap, test_silhouette_features, test_mid_top,
               test_box_location, test_auc):
        fn()
    print("\n" + ("-" * 60))
    print("All checks passed." if not FAILED
          else f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
    raise SystemExit(1 if FAILED else 0)
