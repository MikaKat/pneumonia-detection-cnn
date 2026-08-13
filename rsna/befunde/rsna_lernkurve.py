"""The learning curve of the delivered model, drawn from the per epoch history.

What this produces
------------------
`qc/lernkurve_p5head_ex_de.png` and `..._en.png`, one figure with three panels,
plus the table of numbers behind it on the console. The figure is meant for the
article and for the portfolio, so it exists in both languages.

Why it exists
-------------
Every number this project reports is a single value per fold: one AUC, one loss.
A single value cannot show whether the run was still improving, whether it had
already turned, or how much of the eight epochs was wasted. The training loop
writes a history row after every epoch for exactly this reason, and until now
nothing read those files.

The figure also answers a question an interviewer is likely to ask about a
five model ensemble: which epoch of each run was kept, and how far apart those
epochs are. They are 2, 5, 6, 8 and 8 of eight, so the answer is "far apart",
and that is a property of the curve rather than an accident.

How to read the result
----------------------
Panel 1, discrimination. `sel AUC` is the raw pooled c statistic on the
selection split of 3050 images, the part that is exempt from fitting and from
reporting. It is NOT the stratified AUC (A) that every phase of this project
reports, and it is not comparable with the 0.8687 of the holdout. The dot marks
the epoch whose weights were kept, chosen by this curve and by nothing else.

Panel 2, the pair of losses. Training and selection split, classification only,
so that the two lines measure the same thing. The training loss falls to the
last epoch. The selection loss is lowest in epoch 2 and rises afterwards, which
is the argument for keeping the best epoch rather than the last one.

The two panels disagree, and that is worth saying out loud rather than
smoothing over: the AUC on the same split still creeps upward while the loss on
it is already rising. A ranking can keep improving while the probabilities
attached to it get worse, because AUC reads only the order. This is the reason
the delivered model carries a Platt curve per fold, fitted afterwards on that
same selection split.

Panel 3, the localisation head. The training loss of the second head, which
keeps falling to the last epoch in every fold. A head that sat still here would
be the box trap: a total loss that falls while the head learns nothing.

What would refute the reading: a selection loss that falls monotonically
together with the training loss, or best epochs that all sit at 8. Neither is
the case.

Two traps this script handles rather than describes
---------------------------------------------------
1. `results_rsna.csv` holds SIX rows for the tag `_p5head_ex`. Fold 1 was
   trained twice, the first time with a broken lambda of 7.5e7. Taking the
   first row per fold silently reports the broken run. The last row per fold is
   the one that produced the delivered checkpoint, and the run aborts if a
   selected row carries a lambda outside the bounds the training loop enforces.
2. The training loss in the history is the COMBINED loss of both heads,
   `l_cls + lambda * l_loc`, while the selection loss is classification only.
   Drawing them against each other unchanged would compare two different
   quantities and the gap would be partly the second head. The classification
   part is reconstructed with the lambda of that fold, which is measured once on
   the first batch and constant afterwards.

CLI:
  venv\\Scripts\\python.exe rsna\\befunde\\rsna_lernkurve.py
  venv\\Scripts\\python.exe rsna\\befunde\\rsna_lernkurve.py --sprache en
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _repo_path  # noqa: F401,E402  (puts the neighbour folders on the path)

ARM_TAG = "_p5head_ex"
FOLDS = [0, 1, 2, 3, 4]
SOLL_EPOCHS = 8
LAMBDA_MIN, LAMBDA_MAX = 1e-3, 1e3

# Categorical slots 1 and 2 of the project palette. Validated as a pair:
# CVD separation 24.7, normal vision 33.6, both above the floor.
C_TRAIN = "#000000"
C_SEL = "#000000"
C_FOLD = "#6f6f68"
C_INK = "#000000"
C_MUTED = "#000000"
C_GRID = "#000000"
C_MARKER = "#fff99e"

TEXT = {
    "de": {
        "sup": "Lernkurve des ausgelieferten Modells, fünf Folds, je acht Epochen",
        "t1": "Trennschärfe auf dem Auswahlteil",
        "y1": "AUC (roh, gepoolt)",
        "t2": "Verlust, nur Klassifikation",
        "y2": "Verlust je Bild",
        "t3": "Verlust des zweiten Kopfes",
        "y3": "Lokalisationsverlust (Training)",
        "x": "Epoche",
        "fold": "Fold",
        "mean": "Mittel der fünf Folds",
        "kept": "behaltene Epoche",
        "train": "Training",
        "sel": "Auswahlteil",
        "note1": "Dünn: die fünf Folds. Dick: ihr Mittel. Der Punkt markiert "
                 "die Epoche, deren Gewichte behalten wurden.",
        "note2": "Der Auswahlteil (3050 Bilder) wird weder angepasst noch "
                 "berichtet. Die Zahlen sind nicht die geschichtete Zahl A "
                 "und nicht der Holdout.",
    },
    "en": {
        "sup": "Learning curve of the delivered model, five folds, eight epochs each",
        "t1": "Discrimination on the selection split",
        "y1": "AUC (raw, pooled)",
        "t2": "Loss, classification only",
        "y2": "Loss per image",
        "t3": "Loss of the second head",
        "y3": "Localisation loss (training)",
        "x": "Epoch",
        "fold": "Fold",
        "mean": "Mean of the five folds",
        "kept": "epoch kept",
        "train": "Training",
        "sel": "Selection split",
        "note1": "Thin: the five folds. Thick: their mean. The dot marks the "
                 "epoch whose weights were kept.",
        "note2": "The selection split (3050 images) is neither fitted nor "
                 "reported. These numbers are not the stratified number A, "
                 "and not the holdout.",
    },
}


def abbruch(text: str) -> None:
    print(f"\nABBRUCH: {text}")
    sys.exit(1)


def lies_ergebnisse(pfad: Path, tag: str) -> pd.DataFrame:
    """The last row per fold, with the reason printed rather than assumed."""
    if not pfad.is_file():
        abbruch(f"{pfad} fehlt")
    df = pd.read_csv(pfad)
    arm = df[df["tag"] == tag]
    if arm.empty:
        abbruch(f"{pfad} kennt den Arm {tag!r} nicht")
    print(f"  {pfad}: {len(arm)} Zeilen fuer {tag!r}")
    behalten = []
    for k in FOLDS:
        zeilen = arm[arm["fold"] == k]
        if zeilen.empty:
            abbruch(f"Fold {k} fehlt in {pfad} fuer {tag!r}")
        if len(zeilen) > 1:
            lam = ", ".join(f"{v:.4g}" for v in zeilen["head_lambda"])
            print(f"    Fold {k}: {len(zeilen)} Zeilen (lambda {lam}), "
                  f"die LETZTE gilt")
        behalten.append(zeilen.iloc[-1])
    aus = pd.DataFrame(behalten).reset_index(drop=True)
    schlecht = aus[(aus["head_lambda"] < LAMBDA_MIN) |
                   (aus["head_lambda"] > LAMBDA_MAX)]
    if not schlecht.empty:
        abbruch(f"die gewaehlte Zeile von Fold {int(schlecht.iloc[0]['fold'])} "
                f"hat lambda {schlecht.iloc[0]['head_lambda']:.4g}, ausserhalb "
                f"[{LAMBDA_MIN:g}, {LAMBDA_MAX:g}]. Das ist der kaputte Lauf.")
    return aus


def lies_verlauf(hist_dir: Path, res: pd.DataFrame) -> pd.DataFrame:
    """One frame with the reconstructed classification loss, checked per fold."""
    teile = []
    for _, zeile in res.iterrows():
        k = int(zeile["fold"])
        pfad = hist_dir / f"history_f{k}_s0.csv"
        if not pfad.is_file():
            abbruch(f"{pfad} fehlt. Ohne Verlauf gibt es keine Lernkurve.")
        h = pd.read_csv(pfad)
        if len(h) != SOLL_EPOCHS:
            abbruch(f"{pfad} hat {len(h)} Epochen, erwartet {SOLL_EPOCHS}")
        if int(h["fold"].iloc[0]) != k:
            abbruch(f"{pfad} traegt Fold {int(h['fold'].iloc[0])}, "
                    f"erwartet {k}. Verwechselte Datei.")

        # Die geprueffte Zahl ist nicht die gedruckte: die beste Epoche wird
        # hier aus dem Verlauf NACHGERECHNET und gegen results_rsna.csv
        # gestellt. Weicht sie ab, gehoert der Verlauf zu einem anderen Lauf
        # als der Checkpoint, und die Abbildung waere eine Erfindung.
        best_hist = int(h.loc[h["sel_auc"].idxmax(), "epoch"])
        best_res = int(zeile["best_epoch"])
        if best_hist != best_res:
            abbruch(f"Fold {k}: der Verlauf sagt beste Epoche {best_hist}, "
                    f"results_rsna.csv sagt {best_res}. Der Verlauf gehoert zu "
                    f"einem anderen Lauf.")
        auc_hist = float(h["sel_auc"].max())
        auc_res = float(zeile["auc_sel"])
        if abs(auc_hist - auc_res) > 1e-6:
            abbruch(f"Fold {k}: bestes sel-AUC {auc_hist:.6f} im Verlauf gegen "
                    f"{auc_res:.6f} in results_rsna.csv")

        lam = float(zeile["head_lambda"])
        h["train_cls_loss"] = h["train_loss"] - lam * h["train_loc_loss"]
        if (h["train_cls_loss"] <= 0).any():
            abbruch(f"Fold {k}: der rekonstruierte Klassifikationsverlust wird "
                    f"negativ. lambda {lam:.4g} passt nicht zu diesem Verlauf.")
        h["lambda"] = lam
        h["best_epoch"] = best_hist
        teile.append(h)
    return pd.concat(teile, ignore_index=True)


def tabelle(v: pd.DataFrame) -> None:
    print()
    print("  Fold  lambda  beste Epoche  sel-AUC dort  Verlust Training 1->8  "
          "Verlust Auswahl 1->8")
    for k in FOLDS:
        h = v[v["fold"] == k]
        b = int(h["best_epoch"].iloc[0])
        print(f"  {k:>4}  {h['lambda'].iloc[0]:>6.4f}  {b:>12}  "
              f"{h.loc[h['epoch'] == b, 'sel_auc'].iloc[0]:>12.4f}  "
              f"{h['train_cls_loss'].iloc[0]:>10.4f} -> "
              f"{h['train_cls_loss'].iloc[-1]:<7.4f}  "
              f"{h['sel_loss'].iloc[0]:>8.4f} -> {h['sel_loss'].iloc[-1]:.4f}")
    m = v.groupby("epoch")[["sel_auc", "train_cls_loss", "sel_loss",
                            "train_loc_loss"]].mean()
    print()
    print("  Mittel ueber die fuenf Folds")
    print("  Epoche  sel-AUC  Verlust Training  Verlust Auswahl  Kopf")
    for ep, r in m.iterrows():
        print(f"  {ep:>6}  {r['sel_auc']:>7.4f}  {r['train_cls_loss']:>16.4f}  "
              f"{r['sel_loss']:>15.4f}  {r['train_loc_loss']:.4f}")
    tief = int(m["sel_loss"].idxmin())
    print()
    print(f"  Der mittlere Auswahlverlust ist in Epoche {tief} am tiefsten und "
          f"steigt danach.")
    print(f"  Der Kopfverlust faellt bis zur letzten Epoche "
          f"({m['train_loc_loss'].iloc[0]:.4f} -> "
          f"{m['train_loc_loss'].iloc[-1]:.4f}).")


def achse(ax, t: dict, titel: str, ylab: str) -> None:
    ax.set_title(titel.upper(), fontsize=11.5, fontfamily="Georgia",
                 fontweight="bold", color=C_INK, pad=12, loc="left")
    ax.set_xlabel(t["x"].upper(), fontsize=8.5, fontfamily="Consolas",
                  color=C_MUTED, labelpad=7)
    ax.set_ylabel(ylab.upper(), fontsize=8.5, fontfamily="Consolas",
                  color=C_MUTED, labelpad=9)
    ax.set_xticks(range(1, SOLL_EPOCHS + 1))
    ax.grid(True, color=C_GRID, linewidth=0.45, alpha=0.08, zorder=0)
    ax.set_axisbelow(True)
    for rand in ("top", "right"):
        ax.spines[rand].set_visible(False)
    for rand in ("left", "bottom"):
        ax.spines[rand].set_color(C_INK)
        ax.spines[rand].set_linewidth(1.0)
    ax.tick_params(colors=C_MUTED, labelsize=8.5, length=0)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontfamily("Consolas")


def zeichne(v: pd.DataFrame, sprache: str, ziel: Path) -> None:
    t = TEXT[sprache]
    ep = np.arange(1, SOLL_EPOCHS + 1)
    m = v.groupby("epoch")[["sel_auc", "train_cls_loss", "sel_loss",
                            "train_loc_loss"]].mean()

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.3), dpi=200)
    fig.patch.set_alpha(0)
    for ax in axes:
        ax.set_facecolor("none")

    # Panel 1: sel AUC.
    ax = axes[0]
    achse(ax, t, t["t1"], t["y1"])
    for k in FOLDS:
        h = v[v["fold"] == k]
        ax.plot(ep, h["sel_auc"], color=C_FOLD, linewidth=1.0,
                alpha=0.55, zorder=2)
    ax.plot(ep, m["sel_auc"], color=C_TRAIN, linewidth=2.3, zorder=4,
            label=t["mean"])
    lo, hi = float(v["sel_auc"].min()), float(v["sel_auc"].max())
    ax.set_ylim(lo - 0.14 * (hi - lo), hi + 0.05 * (hi - lo))
    for k in FOLDS:
        h = v[v["fold"] == k]
        b = int(h["best_epoch"].iloc[0])
        y = float(h.loc[h["epoch"] == b, "sel_auc"].iloc[0])
        ax.plot([b], [y], marker="o", markersize=7, color=C_SEL,
                markerfacecolor=C_MARKER, markeredgecolor=C_INK,
                markeredgewidth=1.4, zorder=6,
                linestyle="none",
                label=t["kept"] if k == FOLDS[0] else None)
    ax.plot([], [], color=C_FOLD, linewidth=1.1, label=f"{t['fold']} 0 - 4")
    ax.legend(loc="lower right", prop={"family": "Consolas", "size": 7.5},
              frameon=False, labelcolor=C_MUTED)

    # Panel 2: classification loss, training against selection split.
    ax = axes[1]
    achse(ax, t, t["t2"], t["y2"])
    for k in FOLDS:
        h = v[v["fold"] == k]
        ax.plot(ep, h["train_cls_loss"], color=C_TRAIN, linewidth=0.9,
                alpha=0.18, zorder=2)
        ax.plot(ep, h["sel_loss"], color=C_SEL, linewidth=0.9, alpha=0.18,
                linestyle=(0, (5, 3)),
                zorder=2)
    ax.plot(ep, m["train_cls_loss"], color=C_TRAIN, linewidth=2.3, zorder=4,
            label=t["train"])
    ax.plot(ep, m["sel_loss"], color=C_SEL, linewidth=2.3,
            linestyle=(0, (5, 3)), zorder=4,
            label=t["sel"])
    ax.annotate(t["train"], (ep[-5], m["train_cls_loss"].iloc[-5]),
                xytext=(0, -17), textcoords="offset points", fontsize=8.5,
                fontfamily="Consolas", color=C_MUTED, ha="center")
    ax.annotate(t["sel"], (ep[-1], m["sel_loss"].iloc[-1]),
                xytext=(-2, 9), textcoords="offset points", fontsize=8.5,
                fontfamily="Consolas", color=C_MUTED, ha="right")

    # Panel 3: localisation head.
    ax = axes[2]
    achse(ax, t, t["t3"], t["y3"])
    for k in FOLDS:
        h = v[v["fold"] == k]
        ax.plot(ep, h["train_loc_loss"], color=C_FOLD, linewidth=1.0,
                alpha=0.55, zorder=2)
    ax.plot(ep, m["train_loc_loss"], color=C_TRAIN, linewidth=2.3, zorder=4,
            label=t["mean"])
    ax.plot([], [], color=C_FOLD, linewidth=1.1, label=f"{t['fold']} 0 - 4")
    ax.legend(loc="upper right", prop={"family": "Consolas", "size": 7.5},
              frameon=False, labelcolor=C_MUTED)

    fig.tight_layout(rect=(0.025, 0.012, 0.995, 0.995), h_pad=2.8)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ziel, transparent=True)
    plt.close(fig)
    print(f"  geschrieben: {ziel}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hist-dir", type=Path, default=Path("predictions_final_model"))
    p.add_argument("--results", type=Path, default=Path("results_rsna.csv"))
    p.add_argument("--tag", default=ARM_TAG)
    p.add_argument("--out", type=Path, default=Path("qc") / "lernkurve_p5head_ex")
    p.add_argument("--sprache", choices=["de", "en", "beide"], default="beide")
    args = p.parse_args()

    print("=" * 78)
    print("LERNKURVE DES AUSGELIEFERTEN MODELLS")
    print("=" * 78)

    res = lies_ergebnisse(args.results, args.tag)
    verlauf = lies_verlauf(args.hist_dir, res)
    print(f"  {args.hist_dir}: {len(FOLDS)} Verlaeufe, je {SOLL_EPOCHS} Epochen, "
          f"beste Epoche und sel-AUC gegen {args.results} geprueft")
    tabelle(verlauf)
    print()
    sprachen = ["de", "en"] if args.sprache == "beide" else [args.sprache]
    for s in sprachen:
        zeichne(verlauf, s, args.out.with_name(f"{args.out.name}_{s}.png"))


if __name__ == "__main__":
    main()
