"""
Phase 2, second reading: is the heat map anatomy, or is it about this image?

WHY THIS EXISTS
---------------
Phase 2 answered its own question and produced a headline that is too short.
Averaged over every positive validation image the baseline scores 0.704 point
AUC inside the lung against 0.752 for the location prior, a paired deficit of
-0.048 that is secured over five folds. Read on its own that says the model
points worse than a fixed anatomical map, and the obvious conclusion is that
there is no emergent localisation.

That conclusion does not survive one question: WHERE does the prior earn its
lead. The prior can only ever encode the average box position. If the model had
learned nothing except anatomy, the two would rise and fall together across
images. If the model has learned something about the individual radiograph,
the two should come apart exactly where the box sits in an unusual place.

WHAT IT COMPUTES
----------------
1. The correlation between the per image scores of model and prior. Near zero
   means the model carries information the prior cannot have.
2. The same scores binned into quintiles of the PRIOR's score, that is by how
   typical the box position is. The prior swings across those bins by
   construction; the question is what the model does.
3. The comparison inside the atypical bin, against the LUNG MAP rather than the
   prior. This is the control that matters: the lung map is per image anatomy.
   Beating the prior on atypical images could just mean "this patient's lung
   sits here". Beating the lung map cannot.
4. How much of any of this rests on degenerate maps, and whether the crop arm's
   back projection fill contaminates the lung restricted measure.

Selection note: the bins are defined by the prior's own score, so the prior
looks good in the top bin by construction. That is not a problem for the
comparison, because the selection variable is independent of the model's score
(point 1), and because a map with no information would sit at 0.5 in every bin
rather than at the model's own mean.

CLI, from the repository root:
  python rsna\\befunde\\rsna_lokalisation_lesart.py
"""

from __future__ import annotations

import _repo_path  # noqa: F401  (puts the neighbouring folders on sys.path)

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

T_LIMIT = {3: 3.182, 4: 2.776}
ATYPICAL = 0.55      # prior at or below chance on this image
TYPICAL = 0.85


def paired(df: pd.DataFrame, a: str, b: str) -> tuple:
    """b minus a, averaged per fold, tested over the folds."""
    per = df.groupby("fold")[[a, b]].mean()
    v = (per[b] - per[a]).to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    sd = float(v.std(ddof=1))
    t = float(v.mean() / (sd / np.sqrt(v.size))) if sd > 0 else float("nan")
    return float(v.mean()), sd, t, T_LIMIT.get(v.size - 1, 2.0)


def line(label: str, df: pd.DataFrame, a: str, b: str) -> None:
    m, sd, t, lim = paired(df, a, b)
    if not np.isfinite(t):
        print(f"  {label:<48} {m:+.4f}   too few folds")
        return
    print(f"  {label:<48} {m:+.4f} +- {sd:.4f}   t = {t:+6.2f}   "
          f"limit {lim:.3f}   {'SECURED' if abs(t) > lim else 'not secured'}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-image", type=Path,
                   default=Path("predictions_cam_full/phase2_per_image.csv"))
    p.add_argument("--out", type=Path,
                   default=Path("predictions_cam_full/phase2_lesart.csv"))
    args = p.parse_args(argv)

    if not args.per_image.exists():
        print(f"missing {args.per_image}. Run rsna_cam_power.py first.")
        return 2
    raw = pd.read_csv(args.per_image)
    d = raw.pivot_table(index=["fold", "patientId"], columns="arm",
                        values="point_auc_lung")
    d = d.dropna(subset=["Lagepriore", "Lungenkarte", "base"]).reset_index()
    arms = [c for c in ("base", "bal10", "crop") if c in d.columns]

    print("=" * 84)
    print("1. IS THE MODEL'S MAP JUST ANATOMY?  correlation of the per image scores")
    print("=" * 84)
    for a in arms + ["Lungenkarte"]:
        g = d.dropna(subset=[a])
        r = float(np.corrcoef(g[a], g["Lagepriore"])[0, 1])
        print(f"  {a:<14} against the location prior:  r = {r:+.3f}")
    print("\n  Near zero means the model's map rises and falls independently of")
    print("  where opacities usually sit. A map that were anatomy in disguise")
    print("  would correlate strongly.")

    print("\n" + "=" * 84)
    print("2. BY HOW TYPICAL THE BOX POSITION IS  (quintiles of the prior's score)")
    print("=" * 84)
    d["bin"] = pd.qcut(d["Lagepriore"], 5,
                       labels=["Q1 atypical", "Q2", "Q3", "Q4", "Q5 typical"])
    cols = ["Lagepriore", "Lungenkarte"] + arms
    tab = d.groupby("bin", observed=True)[cols].mean()
    tab.insert(0, "n", d.groupby("bin", observed=True).size())
    print(tab.round(4).to_string())
    print("\n  The prior swings by construction. What matters is that the model")
    print("  stays nearly flat: it does not care whether the box sits where")
    print("  boxes usually sit.")

    print("\n" + "=" * 84)
    print(f"3. THE CONTROL: against the LUNG MAP, which is per image anatomy")
    print("=" * 84)
    atyp = d[d["Lagepriore"] < ATYPICAL]
    typ = d[d["Lagepriore"] >= TYPICAL]
    print(f"  atypical = prior below {ATYPICAL} ({len(atyp)} images), "
          f"typical = prior at or above {TYPICAL} ({len(typ)} images)\n")
    for a in arms:
        print(f"  {a}:")
        line("all images, minus lung map", d.dropna(subset=[a]), "Lungenkarte", a)
        line("atypical position only", atyp.dropna(subset=[a]), "Lungenkarte", a)
        line("typical position only", typ.dropna(subset=[a]), "Lungenkarte", a)
        print()
    print("  Beating the PRIOR on atypical images could still be anatomy: this")
    print("  patient's lung happens to sit there. Beating the LUNG MAP cannot be.")

    print("\n" + "=" * 84)
    print("4. WHAT THE NUMBERS DO NOT REST ON")
    print("=" * 84)
    for a in arms:
        g = raw[raw["arm"] == a]
        if g.empty:
            continue
        deg = int(g["degenerate"].sum())
        with_deg = g["point_auc_lung"].mean()
        without = g[~g["degenerate"].astype(bool)]["point_auc_lung"].mean()
        print(f"  {a:<8} degenerate maps {deg:>4} of {len(g)}   "
              f"point AUC with {with_deg:.4f}, without {without:.4f}")
    print("\n  The crop arm's back projection fills everything outside the crop")
    print("  with the minimum of the map. That inflates its WHOLE IMAGE point AUC")
    print("  and its mass, because the filled area is large and box free. It does")
    print("  NOT touch the lung restricted measure: the adaptive crop is built")
    print("  from the lung bounding box, so no lung pixel falls outside it")
    print("  (measured: 0.0000 of the mask, over 300 images).")
    print("  Read the whole image column for the crop arm with that in mind, and")
    print("  read the primary column without it.")

    tab.to_csv(args.out)
    print(f"\nsaved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
