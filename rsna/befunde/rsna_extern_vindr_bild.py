r"""Reproduzierbare Bloggrafik zur externen VinDr-CXR-Validierung.

Die Grafik wird ausschliesslich aus den gespeicherten Vorhersagen und Berichten
berechnet. Sie enthaelt bewusst keine VinDr-Roentgenbilder: deren
Nutzungsvereinbarung erlaubt keine Weitergabe im Blog.

Aufruf aus dem Repository-Stamm:
  venv\Scripts\python.exe rsna\befunde\rsna_extern_vindr_bild.py \
      --sprache de --out docs\blogpost\assets\06_externe_validierung_vindr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


C_INK = "#111111"
C_MUTED = "#303030"
C_MARK = "#797973"
C_GRID = "#111111"
C_ACCENT = "#fff99e"

TEXT = {
    "de": {
        "prob": "vom Modell gesagte Wahrscheinlichkeit",
        "actual": "tatsächlicher Anteil positiver Bilder",
        "one": "MINDESTENS 1 VON 3",
        "majority": "MINDESTENS 2 VON 3",
        "rank": "AUC der Sortierung",
        "loc": "Punkt-AUC des Lokalisationskopfs",
        "any": "ein Befund genügt, ECE {ece:.3f}",
        "maj": "Mehrheitsregel, ECE {ece:.3f}",
        "ideal": "ideal",
        "raw": "roh",
        "adjusted": "nach Kontrolle der Bildgeometrie",
        "internal": "intern 0.912",
        "prior": "bildunabhängige Lage-Schablone 0.752",
    },
    "en": {
        "prob": "probability predicted by the model",
        "actual": "observed fraction of positive images",
        "one": "1 OF 3",
        "majority": "AT LEAST 2 OF 3",
        "rank": "ranking AUC",
        "loc": "localisation-head point AUC",
        "any": "one finding is enough, ECE {ece:.3f}",
        "maj": "majority rule, ECE {ece:.3f}",
        "ideal": "ideal",
        "raw": "raw",
        "adjusted": "after controlling for image geometry",
        "internal": "internal 0.912",
        "prior": "image-independent position template 0.752",
    },
}


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    kanten = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(p, kanten[1:-1])
    wert = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            wert += m.mean() * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(wert)


def faecher(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    kanten = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(p, kanten[1:-1])
    aus = []
    for b in range(bins):
        m = idx == b
        if m.any():
            aus.append({"n": int(m.sum()), "p": float(p[m].mean()),
                        "y": float(y[m].mean())})
    return aus


def leck_score(d: pd.DataFrame) -> tuple[float, np.ndarray]:
    x = np.column_stack([d.width, d.height, d.width / d.height,
                         d.width * d.height])
    y = d.y.to_numpy(dtype=int)
    oof = np.zeros(len(d), dtype=float)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    for tr, te in cv.split(x, y, d.image_id):
        m = GradientBoostingClassifier(random_state=0).fit(x[tr], y[tr])
        oof[te] = m.predict_proba(x[te])[:, 1]
    return float(roc_auc_score(y, oof)), oof


def leck_bereinigt(y: np.ndarray, p: np.ndarray,
                    leck: np.ndarray, q: int = 5) -> float:
    kanten = np.quantile(leck, np.linspace(0.0, 1.0, q + 1))
    kanten[-1] += 1e-9
    zaehler, nenner = 0.0, 0
    for i in range(q):
        s = (leck >= kanten[i]) & (leck < kanten[i + 1])
        n_pos = int(y[s].sum())
        n_neg = int(s.sum()) - n_pos
        paare = n_pos * n_neg
        if paare == 0:
            continue
        a = float(roc_auc_score(y[s], p[s]))
        zaehler += a * paare
        nenner += paare
    return zaehler / nenner


def achse(ax, xlab: str, ylab: str) -> None:
    ax.set_xlabel(xlab.upper(), fontsize=8.3, fontfamily="Consolas",
                  color=C_MUTED, labelpad=7)
    ax.set_ylabel(ylab.upper(), fontsize=8.3, fontfamily="Consolas",
                  color=C_MUTED, labelpad=9)
    ax.grid(True, color=C_GRID, linewidth=0.45, alpha=0.08, zorder=0)
    ax.set_axisbelow(True)
    for rand in ("top", "right"):
        ax.spines[rand].set_visible(False)
    for rand in ("left", "bottom"):
        ax.spines[rand].set_color(C_INK)
        ax.spines[rand].set_linewidth(1.0)
    ax.tick_params(colors=C_MUTED, labelsize=8.3, length=0)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontfamily("Consolas")


def lade(pfad: Path, lesart: str) -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(pfad / f"extern_vindr_ens_{lesart}.csv")
    b = json.loads((pfad / f"bericht_{lesart}.json").read_text(encoding="utf-8"))
    if len(d) != b["n"] or int(d.y.sum()) != b["positiv"]:
        raise RuntimeError(f"CSV und Bericht {lesart} gehoeren nicht zusammen")
    return d, b


def zeichne(sprache: str, ziel: Path, daten: dict) -> None:
    t = TEXT[sprache]
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 10.4), dpi=200)
    fig.patch.set_alpha(0)
    for ax in axes:
        ax.set_facecolor("none")

    # Kalibrierung: dieselben Modellwerte, zwei unterschiedliche Wahrheiten.
    ax = axes[0]
    achse(ax, t["prob"], t["actual"])
    ax.plot([0, 1], [0, 1], color=C_MARK, linewidth=1.2,
            linestyle=(0, (4, 3)), label=t["ideal"], zorder=2)
    serien = [
        (daten["A"]["bins"], t["any"].format(ece=daten["A"]["ece"]),
         "o", "solid", C_ACCENT),
        (daten["M"]["bins"], t["maj"].format(ece=daten["M"]["ece"]),
         "s", (0, (5, 3)), "none"),
    ]
    for reihe, label, marker, linie, fuellung in serien:
        x = np.array([z["p"] for z in reihe])
        y = np.array([z["y"] for z in reihe])
        n = np.array([z["n"] for z in reihe], dtype=float)
        ax.plot(x, y, color=C_INK, linewidth=2.0, linestyle=linie, zorder=3)
        ax.scatter(x, y, s=18 + 165 * n / n.max(), marker=marker,
                   facecolor=fuellung, edgecolor=C_INK, linewidth=1.2,
                   label=label, zorder=4)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", prop={"family": "Consolas", "size": 7.2},
              frameon=False, labelcolor=C_MUTED)

    # Klassifikation: roh gegen innerhalb der Geometrie-Quintile.
    ax = axes[1]
    achse(ax, "", t["rank"])
    x = np.array([0.0, 1.0])
    roh = np.array([daten["A"]["auc"], daten["M"]["auc"]])
    bereinigt = np.array([daten["A"]["auc_adj"], daten["M"]["auc_adj"]])
    ax.plot(x, roh, color=C_INK, linewidth=1.8, zorder=3)
    ax.scatter(x, roh, s=150, color=C_ACCENT, edgecolor=C_INK,
               linewidth=1.2, label=t["raw"], zorder=4)
    ax.plot(x, bereinigt, color=C_INK, linewidth=1.8,
            linestyle=(0, (5, 3)), zorder=3)
    ax.scatter(x, bereinigt, s=150, facecolor="none", edgecolor=C_INK,
               marker="s", linewidth=1.2, label=t["adjusted"], zorder=4)
    ax.set_xticks(x, [t["one"], t["majority"]])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(0.65, 0.92)
    ax.axhline(0.5, color=C_MARK, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.legend(loc="lower right", prop={"family": "Consolas", "size": 7.2},
              frameon=False, labelcolor=C_MUTED)

    # Lokalisation: vorfestgelegte Lesart und spaetere Mehrheitsregel.
    ax = axes[2]
    achse(ax, "", t["loc"])
    werte = np.array([daten["A"]["kopf"], daten["M"]["kopf"]])
    lo = np.array([daten["A"]["kopf_lo"], daten["M"]["kopf_lo"]])
    hi = np.array([daten["A"]["kopf_hi"], daten["M"]["kopf_hi"]])
    ax.errorbar(x, werte, yerr=np.vstack([werte - lo, hi - werte]),
                fmt="none", ecolor=C_INK, elinewidth=1.5, capsize=5, zorder=3)
    ax.plot(x, werte, color=C_INK, linewidth=1.8, zorder=3)
    ax.scatter(x, werte, s=175, color=C_ACCENT, edgecolor=C_INK,
               linewidth=1.2, zorder=4)
    ax.axhline(0.9123, color=C_INK, linewidth=1.4,
               linestyle=(0, (5, 3)), label=t["internal"])
    ax.axhline(0.7520, color=C_MARK, linewidth=1.4,
               linestyle=(0, (2, 3)), label=t["prior"])
    ax.set_xticks(x, [t["one"], t["majority"]])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(0.64, 0.94)
    ax.legend(loc="lower right", prop={"family": "Consolas", "size": 7.2},
              frameon=False, labelcolor=C_MUTED)

    fig.tight_layout(rect=(0.025, 0.012, 0.995, 0.995), h_pad=2.2)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ziel, transparent=True)
    plt.close(fig)
    print(f"geschrieben: {ziel}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("predictions_extern_vindr"))
    ap.add_argument("--out", type=Path,
                    default=Path("qc") / "extern_vindr")
    ap.add_argument("--sprache", choices=["de", "en", "beide"], default="beide")
    args = ap.parse_args()

    daten = {}
    for lesart in ("A", "M"):
        d, b = lade(args.dir, lesart)
        y = d.y.to_numpy(dtype=int)
        p = d.p_ens.to_numpy(dtype=float)
        leck_auc, leck = leck_score(d)
        daten[lesart] = {
            "auc": float(roc_auc_score(y, p)),
            "auc_adj": float(leck_bereinigt(y, p, leck)),
            "leak": leck_auc,
            "ece": ece(y, p),
            "bins": faecher(y, p),
            "kopf": float(b["kopffeld"]["wert"]),
            "kopf_lo": float(b["kopffeld"]["lo"]),
            "kopf_hi": float(b["kopffeld"]["hi"]),
        }
        print(lesart, {k: round(v, 4) for k, v in daten[lesart].items()
                       if isinstance(v, float)})

    if not np.allclose(pd.read_csv(args.dir / "extern_vindr_ens_A.csv").p_ens,
                       pd.read_csv(args.dir / "extern_vindr_ens_M.csv").p_ens):
        raise RuntimeError("A und M enthalten nicht dieselben Modellwerte")

    sprachen = ["de", "en"] if args.sprache == "beide" else [args.sprache]
    for sprache in sprachen:
        zeichne(sprache, args.out.with_name(args.out.name + f"_{sprache}.png"), daten)


if __name__ == "__main__":
    main()
