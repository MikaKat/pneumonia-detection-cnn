"""Where does the head point when only ONE of three radiologists saw anything?

Why this exists
---------------
The external run failed its primary gate: the head field reached 0.7396 against
a position prior of 0.7520. The breakdown said where the failure sits:

    1 of 3 radiologists agreed   n 750    0.6691
    2 of 3                       n 822    0.7539
    3 of 3                       n 1281   0.8493

So the pointer collapses exactly on the images whose label rests on a single
opinion that two colleagues contradicted. A number cannot settle whether that is
the model missing a subtle opacity or one reader over-calling it. Only a
radiologist looking at the picture can, and that is what this script is for.

THE SELECTION RULE, WRITTEN DOWN BEFORE ANY IMAGE WAS OPENED
------------------------------------------------------------
Otherwise this is a gallery of chosen examples rather than a look at the data.

  1. Only images where EXACTLY ONE radiologist drew a target box (n_rad == 1).
  2. Sort by `point_auc_lung`, ascending.
  3. Take the ten at the quantiles 5, 15, 25 ... 95 percent.

Deterministic, no seed, no peeking. The result spans the whole range from "the
model points somewhere else entirely" to "the model points straight at it", in
their true proportions. Ten worst cases would be as dishonest as ten best.

WHAT EACH PANEL SHOWS
---------------------
Left, the radiograph with every box all three readers drew: the target box of
the one reader who called it, and in a second colour whatever the other two drew
INSTEAD. That second colour is the interesting one. Two readers seeing nothing
is a different situation from two readers seeing cardiomegaly in the same film.

Right, the same picture with the head field over it. Transparent where the field
says nothing, so "no statement" stays visually different from "statement: no",
the same rule the web app follows.

LICENCE, AND IT IS THE REASON THIS SCRIPT EXISTS AT ALL
--------------------------------------------------------
VinDr is PhysioNet Credentialed 1.5.0, "scientific research and no other", and
the Kaggle copy runs under the competition rules. Looking at the images is
exactly the permitted use. PASSING THEM ON IS NOT, and that includes a figure
with a radiograph in it, a crop of one, and an overlay on one.

Therefore this script writes into a folder that it also puts into `.gitignore`,
and it prints what it did. The output is for the screen, never for the repo, the
portfolio or a chat window. The numbers may be reported, the pictures may not.

  venv\\Scripts\\python.exe rsna\\befunde\\rsna_vindr_einzelmeinung.py --dml-index 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _repo_path  # noqa: F401

from rsna_extern_vindr_ens import (FOLDS, ZIEL, baue_kette, feld_auf_referenz,
                                   lade_kalibrierung, lade_modelle,
                                   platt_apply)

QUANTILE = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]


def gitignore_sichern(ordner: Path) -> None:
    """Den Ausgabeordner in .gitignore eintragen, falls er fehlt.

    Nicht Bevormundung, sondern die einzige Stelle, an der ein Lizenzverstoss
    unbemerkt passieren koennte: ein `git add -A` nach diesem Lauf, und die
    Bilder liegen fuer immer in der Geschichte. Git vergisst nichts.
    """
    gi = Path(".gitignore")
    eintrag = f"{ordner.as_posix()}/"
    vorhanden = gi.read_text(encoding="utf-8") if gi.is_file() else ""
    if eintrag in vorhanden:
        print(f"  .gitignore kennt {eintrag} bereits.")
        return
    with open(gi, "a", encoding="utf-8") as f:
        f.write(f"\n# VinDr-Bilder duerfen nicht weitergegeben werden "
                f"(PhysioNet Credentialed 1.5.0)\n{eintrag}\n")
    print(f"  {eintrag} in .gitignore eingetragen.")


def auswahl(kopf_csv: Path) -> pd.DataFrame:
    """Die oben aufgeschriebene Regel, und nichts sonst."""
    k = pd.read_csv(kopf_csv)
    einer = k[k.n_rad == 1]
    if einer.empty:
        raise SystemExit(f"{kopf_csv} enthaelt keine Einzelmeinungen. "
                         f"Erst rsna_extern_vindr_ens.py --lesart A laufen lassen.")
    spalte = "point_auc_lung" if einer.point_auc_lung.notna().any() else "point_auc"
    je_bild = (einer.groupby("image_id")
                    .agg(wert=(spalte, "mean"), rad=("rad_id", "first"))
                    .sort_values("wert"))
    idx = [int(round(q * (len(je_bild) - 1))) for q in QUANTILE]
    aus = je_bild.iloc[idx].copy()
    aus["quantil"] = QUANTILE
    aus["spalte"] = spalte
    return aus


def zeichne(ax, bild, boxen_ziel, boxen_andere, feld=None, alpha=0.55):
    from matplotlib.patches import Rectangle

    ax.imshow(bild, cmap="gray", vmin=0, vmax=255)
    if feld is not None:
        # Durchsichtig, wo das Feld nichts sagt. Dieselbe Regel wie in der App:
        # der Unterschied zwischen "keine Aussage" und "Aussage: nein".
        m = np.ma.masked_where(feld < 0.15, feld)
        ax.imshow(m, cmap="inferno", alpha=alpha, vmin=0, vmax=1,
                  extent=[0, bild.shape[1], bild.shape[0], 0])
    for (x, y, w, h) in boxen_ziel:
        ax.add_patch(Rectangle((x, y), w, h, fill=False,
                               edgecolor="#22d3ee", lw=2.0))
    for name, (x, y, w, h) in boxen_andere:
        ax.add_patch(Rectangle((x, y), w, h, fill=False,
                               edgecolor="#f59e0b", lw=1.2, ls="--"))
        ax.text(x, max(y - 6, 10), name, color="#f59e0b", fontsize=7)
    ax.set_xticks([]), ax.set_yticks([])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bilder", type=Path,
                    default=Path("data/vinbigdata/vinbigdata/train"))
    ap.add_argument("--csv", type=Path,
                    default=Path("data/vinbigdata/vinbigdata/train.csv"))
    ap.add_argument("--kopf", type=Path,
                    default=Path("predictions_extern_vindr/extern_vindr_kopf_A.csv"))
    ap.add_argument("--out", type=Path, default=Path("qc/vindr_einzelmeinung"))
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--dml-index", type=int, default=None)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image

    print("=" * 74)
    print("EINZELMEINUNGEN: wohin zeigt der Kopf, wenn nur EINER etwas sah?")
    print("=" * 74)
    a.out.mkdir(parents=True, exist_ok=True)
    gitignore_sichern(a.out)

    aus = auswahl(a.kopf)
    print(f"\n  Auswahlregel: n_rad == 1, nach {aus.spalte.iloc[0]} sortiert,")
    print(f"  die zehn an den Quantilen {', '.join(f'{q:.0%}' for q in QUANTILE)}.")
    print(f"  Spanne der Auswahl: {aus.wert.min():.4f} bis {aus.wert.max():.4f}")

    d = pd.read_csv(a.csv)
    ziel = d[d.class_name.isin(ZIEL)].dropna(subset=["x_min"])
    alle = d.dropna(subset=["x_min"])
    masse = d.groupby("image_id")[["width", "height"]].first()

    device = (__import__("torch_directml").device(a.dml_index)
              if a.dml_index is not None
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    _, kurven, schwelle = lade_kalibrierung(
        Path("serving/model/kalibrierung_p10.json"))
    print()
    modelle, _ = lade_modelle(device)
    tf = baue_kette(a.size)

    print(f"\n  {'#':>2} {'Bild':<14}{'Quantil':>9}{'Zeiger':>9}{'p':>8}"
          f"  {'Ruf':<5} was die anderen zwei sahen")
    print("  " + "-" * 86)

    for i, (iid, r) in enumerate(aus.iterrows(), 1):
        pfad = a.bilder / f"{iid}.png"
        bild = np.array(Image.open(pfad).convert("L"))
        H, W = bild.shape
        w0, h0 = float(masse.loc[iid, "width"]), float(masse.loc[iid, "height"])

        with torch.no_grad():
            x = tf(Image.open(pfad).convert("L"))[None].to(device)
            ps, felder = [], []
            for kf, m in zip(FOLDS, modelle):
                lo, fe = m(x)
                ps.append(platt_apply(
                    torch.sigmoid(lo.squeeze(1)).float().cpu().numpy(), *kurven[kf]))
                felder.append(torch.sigmoid(fe[:, 0]).float().cpu().numpy()[0])
        p = float(np.mean(ps))
        feld = feld_auf_referenz(np.mean(felder, axis=0))

        # Kaesten aus Originalkoordinaten ins Anzeigeraster des 512er-PNG.
        # NICHT ueber BOX_SPACE: das 1024er Raster ist der Weg fuer die MESSUNG,
        # damit sie neben den RSNA-Zahlen steht. Hier wird nur gezeichnet, und
        # dafuer ist der direkte Weg der richtige und der kuerzere.
        sw, sh = W / w0, H / h0
        zb = [(float(t.x_min) * sw, float(t.y_min) * sh,
               float(t.x_max - t.x_min) * sw, float(t.y_max - t.y_min) * sh)
              for t in ziel[ziel.image_id == iid].itertuples()]
        rad_ziel = {t.rad_id for t in ziel[ziel.image_id == iid].itertuples()}
        andere = [(t.class_name, (float(t.x_min) * sw, float(t.y_min) * sh,
                                  float(t.x_max - t.x_min) * sw,
                                  float(t.y_max - t.y_min) * sh))
                  for t in alle[alle.image_id == iid].itertuples()
                  if t.rad_id not in rad_ziel]
        was_andere = (", ".join(sorted({n for n, _ in andere})) or
                      "NICHTS (beide sagten: kein Befund)")

        fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
        zeichne(axes[0], bild, zb, andere)
        axes[0].set_title("Kaesten der Radiologen", fontsize=10)
        zeichne(axes[1], bild, zb, andere, feld=feld)
        axes[1].set_title("dazu das Kopffeld", fontsize=10)
        fig.suptitle(
            f"{i}. {iid[:12]}   Quantil {r.quantil:.0%}   "
            f"{aus.spalte.iloc[0]} {r.wert:.3f}   Modell {p:.1%}"
            f"   (Schwelle {schwelle:.1%})\n"
            f"tuerkis = der EINE Radiologe ({r.rad}), der eine Verschattung sah   |   "
            f"orange = was die anderen zwei sahen: {was_andere}",
            fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig.savefig(a.out / f"{i:02d}_{iid[:12]}.png", dpi=130)
        plt.close(fig)

        print(f"  {i:>2} {iid[:12]:<14}{r.quantil:>8.0%}{r.wert:>9.3f}{p:>8.1%}"
              f"  {r.rad:<5} {was_andere[:44]}")

    print(f"\n  -> {a.out}  (zehn Abbildungen)")
    print("\n  ZUM ANSEHEN, NICHT ZUM WEITERGEBEN. Die Bilder bleiben auf diesem")
    print("  Rechner: nicht ins Repo, nicht in die Mappe, nicht in einen Chat.")
    print("  Berichtet werden duerfen die Zahlen, nicht die Roentgenbilder.")


if __name__ == "__main__":
    main()
