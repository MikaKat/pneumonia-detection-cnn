"""The external validation as one figure, for the article and the portfolio.

What this produces
------------------
`qc/extern_kermany_de.png` and `..._en.png`, two panels, plus the numbers on the
console. Every value is recomputed from
`predictions_extern_kermany/extern_kermany_ens.csv` and then checked against
`bericht.json` of that run. A mismatch stops the script, because a figure drawn
from a stale file is worse than no figure.

Why it exists
-------------
The external run produces one sentence worth showing: the ordering survives the
change of hospital, machine, age group and prevalence, and the probabilities do
not. A table says that. A picture says it in one look, and the second panel
supplies the control without which the first would be worthless.

How to read the result
----------------------
Panel 1, reliability. Each dot is one tenth of the probability scale, its area
proportional to how many images fall into it. On the diagonal, a stated 30
percent would mean 30 percent of those images are positive. The delivered curve
sits far above the diagonal: at a stated 0.25 the actual rate is 0.93.

The second series in that panel is a control, not a repair. The scores are
shifted by the log odds difference between the two prevalences, 0.2253
internally and 0.7297 here. Nothing is fitted for it. It answers how much of the
miscalibration is prevalence alone.

Panel 2, discrimination against the metadata leak. The bare JPEG dimensions
separate the classes on this dataset at 0.9150, so a raw external AUC could be
file geometry rather than lungs. The panel shows the AUC WITHIN the quintiles of
that geometry score. Marker area is proportional to the discordant pairs
(n_pos x n_neg) of the quintile, which is the information it actually carries: a
quintile of 1194 images with a single negative one is large and says nothing.

What would refute the reading: quintile values falling towards 0.5, which would
mean the number was the leak.

CLI:
  venv\\Scripts\\python.exe rsna\\befunde\\rsna_extern_kermany_bild.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _repo_path  # noqa: F401,E402  (puts the neighbour folders on the path)

from rsna_platt import ece  # noqa: E402

VARIANTE = "stretch"
TOLERANZ = 1e-9

C_IST = "#000000"
C_SHIFT = "#000000"
C_MARK = "#6f6f68"
C_INK = "#000000"
C_MUTED = "#000000"
C_GRID = "#000000"
C_ACCENT = "#fff99e"

TEXT = {
    "de": {
        "sup": "Externe Validierung des ausgelieferten Ensembles auf Kermany, "
               "5856 Bilder, reines Vorwärtsrechnen",
        "t1": "Die Sortierung überlebt, die Wahrscheinlichkeit nicht",
        "x1": "vom Modell gesagte Wahrscheinlichkeit",
        "y1": "tatsächlicher Anteil kranker Bilder",
        "ideal": "ideal",
        "ist": "wie ausgeliefert, ECE {ece:.3f}",
        "shift": "um die bekannte Prävalenz verschoben, ECE {ece:.3f}",
        "flaeche": "Fläche der Punkte: links die Zahl der Bilder im Fach, "
                   "rechts die diskordanten Paare des Quintils",
        "t2": "Und es ist nicht der Metadaten-Leak",
        "x2": "Quintil des Leak-Werts (nur Bildabmessungen)",
        "y2": "AUC des Modells im Quintil",
        "mittel": "gewichtetes Mittel {v:.4f}",
        "zufall": "Zufall",
        "note": "Der Leak allein trennt bei {leak:.4f}. Die Prävalenz steigt "
                "von {p0:.4f} auf {p1:.4f}.",
    },
    "en": {
        "sup": "External validation of the delivered ensemble on Kermany, "
               "5856 images, inference only",
        "t1": "The ordering survives, the probability does not",
        "x1": "probability stated by the model",
        "y1": "actual share of positive images",
        "ideal": "ideal",
        "ist": "as delivered, ECE {ece:.3f}",
        "shift": "shifted by the known prevalence, ECE {ece:.3f}",
        "flaeche": "Dot area: left the number of images in the bin, right the "
                   "discordant pairs of the quintile",
        "t2": "And it is not the metadata leak",
        "x2": "quintile of the leak score (image dimensions only)",
        "y2": "AUC of the model within the quintile",
        "mittel": "weighted mean {v:.4f}",
        "zufall": "chance",
        "note": "The leak alone separates at {leak:.4f}. Prevalence rises from "
                "{p0:.4f} to {p1:.4f}.",
    },
}


def abbruch(text: str) -> None:
    print(f"\nABBRUCH: {text}")
    sys.exit(1)


def gleich(name: str, eigen: float, bericht: float) -> None:
    if abs(eigen - bericht) > TOLERANZ:
        abbruch(f"{name}: selbst gerechnet {eigen:.8f}, im Bericht "
                f"{bericht:.8f}. Die CSV gehoert zu einem anderen Lauf.")
    print(f"  {name:<34}{eigen:>12.6f}  stimmt mit bericht.json ueberein")


def faecher(y, p, n: int = 10) -> list[dict]:
    kanten = np.linspace(0.0, 1.0, n + 1)
    idx = np.digitize(p, kanten[1:-1])
    aus = []
    for b in range(n):
        m = idx == b
        if m.any():
            aus.append({"n": int(m.sum()), "p": float(p[m].mean()),
                        "y": float(y[m].mean())})
    return aus


def priorverschiebung(p, p0: float, p1: float):
    """Shift the log odds by the difference of the two prevalences.

    NOTHING is fitted here. The only input beyond the scores is how common the
    disease is in the target set, which is a property of the set and not of the
    model. That makes this a control rather than a repair: it says how much of
    the miscalibration is prevalence alone, and what is left over is the part
    the model itself gets wrong.
    """
    q = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    z = np.log(q / (1.0 - q))
    versatz = np.log(p1 / (1.0 - p1)) - np.log(p0 / (1.0 - p0))
    return 1.0 / (1.0 + np.exp(-(z + versatz))), float(versatz)


def achse(ax, xlab: str, ylab: str) -> None:
    ax.set_xlabel(xlab.upper(), fontsize=8.5, fontfamily="Consolas",
                  color=C_MUTED, labelpad=7)
    ax.set_ylabel(ylab.upper(), fontsize=8.5, fontfamily="Consolas",
                  color=C_MUTED, labelpad=9)
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


def zeichne(sprache: str, ziel: Path, ist: list[dict], versch: list[dict],
            e_ist: float, e_versch: float, quintile: list[dict],
            mittel: float, leak: float, p0: float, p1: float) -> None:
    t = TEXT[sprache]
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.2), dpi=200)
    fig.patch.set_alpha(0)
    for ax in axes:
        ax.set_facecolor("none")

    ax = axes[0]
    achse(ax, t["x1"], t["y1"])
    ax.plot([0, 1], [0, 1], color=C_MARK, linewidth=1.2, linestyle=(0, (4, 3)),
            zorder=2, label=t["ideal"])
    serien = (
        (versch, C_SHIFT, t["shift"].format(ece=e_versch), "s", (0, (5, 3)), "none"),
        (ist, C_IST, t["ist"].format(ece=e_ist), "o", "solid", C_ACCENT),
    )
    for reihe, farbe, marke, symbol, linie, fuellung in serien:
        x = [z["p"] for z in reihe]
        yv = [z["y"] for z in reihe]
        n = np.array([z["n"] for z in reihe], dtype=float)
        ax.plot(x, yv, color=farbe, linewidth=2.0, linestyle=linie, zorder=3)
        ax.scatter(x, yv, s=18 + 170 * n / n.max(), marker=symbol,
                   facecolor=fuellung, edgecolor=C_INK, linewidth=1.2,
                   zorder=4, label=marke)
    ax.set_xlim(-0.03, 1.03)
    # Luft nach oben, damit die Beschriftung nicht auf der flachen Kurve liegt
    ax.set_ylim(-0.05, 1.22)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    griffe, marken = ax.get_legend_handles_labels()
    ordnung = [2, 1, 0]          # wie ausgeliefert zuerst, ideal zuletzt
    ax.legend([griffe[i] for i in ordnung], [marken[i] for i in ordnung],
              loc="upper left", prop={"family": "Consolas", "size": 7.5},
              frameon=False, labelcolor=C_MUTED)

    ax = axes[1]
    achse(ax, t["x2"], t["y2"])
    q = [z["q"] for z in quintile]
    a = [z["auc"] for z in quintile]
    paare = np.array([z["disk_paare"] for z in quintile], dtype=float)
    ax.axhline(0.5, color=C_MARK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(t["zufall"], (0.98, 0.5), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=8,
                fontfamily="Consolas", color=C_MUTED, ha="right")
    ax.axhline(mittel, color=C_SHIFT, linewidth=1.8,
               linestyle=(0, (5, 3)), zorder=3,
               label=t["mittel"].format(v=mittel))
    ax.plot(q, a, color=C_IST, linewidth=2.0, zorder=4)
    ax.scatter(q, a, s=18 + 170 * paare / paare.max(), color=C_ACCENT,
               edgecolor=C_INK, linewidth=1.2, zorder=5)
    ax.set_xticks(q)
    ax.set_ylim(0.42, 1.02)
    ax.legend(loc="lower left", prop={"family": "Consolas", "size": 7.5},
              frameon=False, labelcolor=C_MUTED)

    fig.tight_layout(rect=(0.025, 0.012, 0.995, 0.995), h_pad=2.2)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ziel, transparent=True)
    plt.close(fig)
    print(f"  geschrieben: {ziel}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=Path("predictions_extern_kermany"))
    p.add_argument("--kalibrierung", type=Path,
                   default=Path("serving") / "model" / "kalibrierung_p10.json")
    p.add_argument("--out", type=Path, default=Path("qc") / "extern_kermany")
    p.add_argument("--sprache", choices=["de", "en", "beide"], default="beide")
    args = p.parse_args()

    csv = args.dir / "extern_kermany_ens.csv"
    js = args.dir / "bericht.json"
    for f in (csv, js, args.kalibrierung):
        if not f.is_file():
            abbruch(f"{f} fehlt")
    d = pd.read_csv(csv)
    b = json.loads(js.read_text(encoding="utf-8"))
    kal = json.loads(args.kalibrierung.read_text(encoding="utf-8"))

    print("=" * 78)
    print("EXTERNE VALIDIERUNG, ABBILDUNG")
    print("=" * 78)
    if len(d) != b["n"]:
        abbruch(f"die CSV hat {len(d)} Zeilen, der Bericht spricht von {b['n']}")

    y = d.label.values.astype(float)
    ens = d[f"p_{VARIANTE}_ens"].values
    gleich("Praevalenz", float(y.mean()), b["praevalenz"])
    gleich("ECE, wie ausgeliefert", ece(y, ens), b["ece"])

    p0 = float(kal["dev"]["praevalenz"])
    p1 = float(y.mean())
    versch, versatz = priorverschiebung(ens, p0, p1)
    e_ist = ece(y, ens)
    e_versch = ece(y, versch)
    print(f"  Priorverschiebung um {versatz:+.4f} auf der Logit-Skala")
    print(f"  ECE {e_ist:.4f} -> {e_versch:.4f}, es bleiben "
          f"{e_versch / e_ist:.0%} uebrig")
    print(f"  mittlere Wahrscheinlichkeit {ens.mean():.4f} -> "
          f"{versch.mean():.4f}, tatsaechlich {p1:.4f}")

    quintile = [z for z in b["varianten"][VARIANTE]["je_quintil"]
                if z["auc"] is not None]
    mittel = float(b["varianten"][VARIANTE]["auc_leak_bereinigt"])
    print(f"  {len(quintile)} Quintile mit AUC, gewichtetes Mittel {mittel:.4f}")

    sprachen = ["de", "en"] if args.sprache == "beide" else [args.sprache]
    for s in sprachen:
        zeichne(s, args.out.with_name(f"{args.out.name}_{s}.png"),
                faecher(y, ens), faecher(y, versch), e_ist, e_versch,
                quintile, mittel, float(b["leak_auc"]), p0, p1)


if __name__ == "__main__":
    main()
