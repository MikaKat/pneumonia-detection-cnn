"""
Phase 6: the verdict on the augmentation arm, recomputed from scratch.

WHY THIS SCRIPT EXISTS AT ALL
-----------------------------
The training run already prints an AUC per fold. This script does not trust
those lines. It recomputes both endpoints from the per image prediction files,
compares the result against the numbers in `results_rsna.csv`, and refuses to
print a verdict if the two disagree. That is the standing rule of this project:
the conclusions of a phase get recomputed by a script that did not produce
them.

It also refuses to run if the arm did not carry the strength it claims. The
augmentation strength lives in four columns of `results_rsna.csv`. If the
phase 6 rows do not read 0.08 / 0.75 / 1.00, the comparison would silently be
between two identical recipes, and the answer would look like a null result
instead of a mistake.

THE PRE-REGISTRATION, COPIED FROM erklaerungen/20_phase6_vorfestlegung.md
-------------------------------------------------------------------------
Written before the run. Anchors are the five fold means of the phase 5 winner
`predictions_final_model`:

    A = 0.8368    stratified AUC
    C = 0.7467    AUC(score -> ViewPosition)

PRIMARY, C FALLS. Paired per fold against the anchor, 90 percent interval of
    the difference, the UPPER end must lie below zero.

CONSTRAINT, A IS NON INFERIOR. Margin 0.01, paired per fold, 90 percent
    interval, the LOWER end must lie above -0.01. Same margin as phase 5
    and phase 8.

BOTH must hold. The old roadmap gate "C must fall from 0.8166" is not used:
0.8166 is the phase 0 baseline WITHOUT reweighting, every arm since phase 5
starts below it, and a gate that saturates decides nothing.

WHAT THE SECOND HALF IS, AND WHAT IT IS NOT
-------------------------------------------
Everything after the verdict is DESCRIPTIVE. It exists because a failed
primary endpoint is only useful if one can say what failed. Two diagnostics:

  * C under each test time perturbation, both arms. If the framing carried
    the projection signal, masking the corners or shifting the image should
    lower C. It does not.

  * How much of the geometric cue the augmentation actually removed,
    simulated on the project's own geometry table. This is the question the
    verdict cannot answer: a knob that was never turned far enough did not
    test its hypothesis, it only failed to.

None of this changes the verdict above, and no threshold in it was chosen
after seeing a number.

USAGE
-----
    python rsna\\befunde\\rsna_phase6_auswertung.py
    python rsna\\befunde\\rsna_phase6_auswertung.py --nur-urteil
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _repo_path  # noqa: F401  (puts the neighbouring folders on sys.path)

# --------------------------------------------------------------------------
# The pre-registration, as constants. Changing one of these numbers is
# changing the experiment, and it should look like that in a diff.
# --------------------------------------------------------------------------
ANKER_A = 0.8368
ANKER_C = 0.7467
MARGE_A = 0.01
NIVEAU = 0.90            # two sided, so the one sided bound is at 5 percent

ARM_TAG = "_p6aug"
ARM_DIR = "predictions_p6_aug"
BEZUG_TAG = "_p5head_ex"
BEZUG_DIR = "predictions_final_model"

# The strength the arm claims to have trained with. Checked, not assumed.
SOLL_AUG = dict(aug_translate=0.08, aug_scale_lo=0.75, aug_scale_hi=1.00)
# The reference arm predates the switches, so its columns are empty. If they
# are filled, they must read the old hard wired values.
ALT_AUG = dict(aug_translate=0.03, aug_scale_lo=0.93, aug_scale_hi=1.07)

ANKER_TOLERANZ = 5e-5    # the anchors are quoted to four places
HERKUNFT_TOLERANZ = 1e-9  # recomputation against results_rsna.csv

STOERUNGEN = ["p_clean", "p_corners", "p_zoom_in", "p_shift", "p_rotate",
              "p_low_contr", "p_bright", "p_blur", "p_lowres"]
GEOMETRISCH = {"p_corners", "p_zoom_in", "p_shift"}


# --------------------------------------------------------------------------
# Statistics, written here rather than imported, so that a second
# implementation exists
# --------------------------------------------------------------------------

def rank_auc(score, label) -> float:
    """AUC as the rank statistic. Ties get their mean rank, which is what
    makes this agree with the trapezoid reading of the ROC curve."""
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    if label.all() or not label.any():
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    n1 = int(label.sum())
    n0 = int((~label).sum())
    return float((r[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def stratified_auc(score, label, view) -> float:
    """The n weighted mean of the AP only and the PA only AUC.

    Pooling the two projections rewards a model for telling AP from PA, which
    is the confounder itself. This project never reports the pooled number
    alone, see the note on pooled measures in the memory.
    """
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    view = np.asarray(view)
    teile, gewichte = [], []
    for v in ("AP", "PA"):
        m = view == v
        a = rank_auc(score[m], label[m])
        if np.isfinite(a):
            teile.append(a)
            gewichte.append(int(m.sum()))
    if not teile:
        return float("nan")
    return float(np.average(teile, weights=gewichte))


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-16:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    from math import exp, lgamma, log
    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b)
             + a * log(x) + b * log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    x = df / (df + t * t)
    p = 0.5 * _betai(df / 2.0, 0.5, x)
    return 1.0 - p if t > 0 else p


def t_ppf(p: float, df: int) -> float:
    """Bisection on t_cdf. scipy is not a dependency of this repo, and a
    quantile that only exists inside scipy cannot be checked by reading it."""
    lo, hi = -60.0, 60.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gepaart(neu: np.ndarray, alt: np.ndarray, niveau: float = NIVEAU) -> dict:
    """Paired per fold: neu minus alt, one difference per fold.

    Paired means the same fold compared with itself. The spread BETWEEN folds
    is larger than the effect being looked for, so an unpaired comparison
    would drown it.
    """
    v = np.asarray(neu, dtype=float) - np.asarray(alt, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return {"n": n, "mean": float(v.mean()) if n else float("nan"),
                "sd": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "je_fold": v}
    sd = float(v.std(ddof=1))
    se = sd / np.sqrt(n)
    h = t_ppf(0.5 + niveau / 2.0, n - 1) * se
    return {"n": n, "mean": float(v.mean()), "sd": sd,
            "lo": float(v.mean() - h), "hi": float(v.mean() + h),
            "je_fold": v}


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def zeilen_holen(res: pd.DataFrame, tag: str) -> pd.DataFrame:
    d = res[res["tag"] == tag]
    if d.empty:
        abbruch(f"kein Arm mit tag '{tag}' in results_rsna.csv")
    doppelt = d["fold"].duplicated().sum()
    if doppelt:
        print(f"  Hinweis: {doppelt} Fold(s) von '{tag}' stehen mehrfach in "
              f"results_rsna.csv, es zaehlt der letzte Eintrag.")
        d = d.drop_duplicates(subset="fold", keep="last")
    return d.sort_values("fold").reset_index(drop=True)


def abbruch(text: str) -> None:
    print(f"\nABBRUCH: {text}")
    sys.exit(1)


def staerke_pruefen(d: pd.DataFrame, tag: str, soll: dict,
                    darf_leer: bool) -> None:
    """The check that stops the most expensive mistake of this phase.

    An arm that ran with the old augmentation would produce numbers that look
    like a clean null result. Nothing in the output would say so. The strength
    is in the result file for exactly this reason.
    """
    for spalte, wert in soll.items():
        if spalte not in d.columns:
            if darf_leer:
                continue
            abbruch(f"'{tag}' hat keine Spalte {spalte}. Der Lauf stammt aus "
                    f"einer Fassung vor den Augmentierungsschaltern.")
        ist = d[spalte].to_numpy(dtype=float)
        if np.all(np.isnan(ist)):
            if darf_leer:
                continue
            abbruch(f"'{tag}' hat {spalte} leer. Die Staerke ist nicht belegt.")
        if not np.allclose(ist, wert, atol=1e-9, equal_nan=False):
            abbruch(f"'{tag}' hat {spalte} = {sorted(set(ist))}, erwartet "
                    f"{wert}. Dieser Arm trainierte NICHT mit der Staerke, "
                    f"gegen die hier geurteilt wird.")


def nachrechnen(pred_dir: Path, folds) -> pd.DataFrame:
    """Both endpoints, recomputed from the per image files."""
    zeilen = []
    for f in folds:
        p = pred_dir / f"rsna_f{f}_s0.csv"
        if not p.exists():
            abbruch(f"{p} fehlt. Ohne die Vorhersagen je Bild ist keine "
                    f"unabhaengige Nachrechnung moeglich.")
        x = pd.read_csv(p)
        ap = (x["viewpos"] == "AP").to_numpy()
        y = x["y"].to_numpy(dtype=bool)
        zeile = {"fold": f, "n": len(x),
                 "A": stratified_auc(x["p_clean"], y, x["viewpos"])}
        for s in STOERUNGEN:
            zeile["C_" + s[2:]] = rank_auc(x[s], ap) if s in x.columns else np.nan
        zeile["C"] = zeile["C_clean"]
        zeilen.append(zeile)
    return pd.DataFrame(zeilen)


def herkunft(name: str, res: pd.DataFrame, neu: pd.DataFrame) -> None:
    for spalte, gerechnet in (("auc_stratified", "A"), ("auc_view", "C")):
        soll = res[spalte].to_numpy(dtype=float)
        ist = neu[gerechnet].to_numpy(dtype=float)
        ab = float(np.abs(soll - ist).max())
        zeichen = "ok  " if ab <= HERKUNFT_TOLERANZ else "FEHLER"
        print(f"  {zeichen} {name:<16} {spalte:<16} groesste Abweichung "
              f"{ab:.2e}")
        if ab > HERKUNFT_TOLERANZ:
            abbruch(f"{name}: {spalte} laesst sich aus den Vorhersagen nicht "
                    f"reproduzieren. Entweder passen Datei und Zeile nicht "
                    f"zusammen, oder eine der beiden ist veraltet.")


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------

def urteil_c(r: dict) -> tuple[str, str]:
    """C must FALL. A rise is the problem, so the UPPER end decides."""
    if r["hi"] < 0:
        return "BESTANDEN", "das obere Ende liegt unter null, C faellt"
    if r["lo"] > 0:
        return "DURCHGEFALLEN", "C ist gesichert GESTIEGEN"
    return "DURCHGEFALLEN", ("das obere Ende liegt ueber null, ein Anstieg "
                             "ist nicht ausgeschlossen")


def urteil_a(r: dict) -> tuple[str, str]:
    """A must not fall by more than the margin. The LOWER end decides."""
    if r["lo"] > -MARGE_A:
        return "BESTANDEN", (f"das untere Ende liegt ueber "
                             f"{-MARGE_A:+.2f}, A ist nicht unterlegen")
    return "DURCHGEFALLEN", (f"das untere Ende liegt unter {-MARGE_A:+.2f}, "
                             f"ein Verlust ueber der Marge ist moeglich")


def block(titel: str) -> None:
    print("\n" + "=" * 78)
    print(titel)
    print("=" * 78)


def zeige(name: str, r: dict, spruch: tuple[str, str]) -> None:
    print(f"\n  {name}")
    print("    je Fold  " + "  ".join(f"{x:+.4f}" for x in r["je_fold"]))
    print(f"    Differenz {r['mean']:+.4f} +- {r['sd']:.4f}   "
          f"{int(NIVEAU * 100)}-Prozent-Intervall "
          f"[{r['lo']:+.4f}, {r['hi']:+.4f}]")
    print(f"    {spruch[0]}: {spruch[1]}")


# --------------------------------------------------------------------------
# Descriptive part
# --------------------------------------------------------------------------

def stoerungen_zeigen(bez: pd.DataFrame, arm: pd.DataFrame) -> None:
    block("BESCHREIBEND 1: C unter Stoerung des Testbildes")
    print("  Nicht vorfestgelegt. Die Frage ist, WORIN der Projektionshinweis")
    print("  steckt. Wuerde ihn die Rahmung tragen, muesste C fallen, sobald")
    print("  man die Ecken abdeckt, zoomt oder verschiebt.")
    print()
    print(f"  {'Stoerung':<12}{'Phase 5':>10}{'Phase 6':>10}"
          f"{'P5 gg clean':>13}{'P6 gg clean':>13}")
    b0 = bez["C_clean"].mean()
    a0 = arm["C_clean"].mean()
    for s in STOERUNGEN:
        k = "C_" + s[2:]
        b, a = bez[k].mean(), arm[k].mean()
        mark = "  <-- Rahmung" if s in GEOMETRISCH else ""
        print(f"  {s[2:]:<12}{b:>10.4f}{a:>10.4f}{b - b0:>+13.4f}"
              f"{a - a0:>+13.4f}{mark}")


def geometrie_simulieren(pfad: Path, ziehungen: int = 200) -> None:
    block("BESCHREIBEND 2: wie viel vom Groessenhinweis nahm die "
          "Augmentierung?")
    if not pfad.exists():
        print(f"  {pfad} fehlt, uebersprungen.")
        return
    g = pd.read_csv(pfad)
    if not {"vp", "width", "aspect"} <= set(g.columns):
        print(f"  {pfad} hat nicht die erwarteten Spalten, uebersprungen.")
        return
    ap = (g["vp"] == "AP").to_numpy()
    rng = np.random.default_rng(0)

    def abstand(x) -> float:
        return abs(rank_auc(x, ap) - 0.5)

    print("  Simuliert auf der eigenen Geometrietabelle. RandomAffine zieht")
    print("  je Bild EINE Skala; die Breite geht linear mit, die Flaeche mit")
    print("  dem Quadrat. Gemessen wird der Abstand zur Muenze, weil die")
    print("  Richtung hier nichts bedeutet.")
    print()
    print(f"  {'Groesse':<10}{'ohne Aug':>10}{'Phase 5':>12}{'Phase 6':>12}"
          f"{'P6 uebrig':>12}")
    for sp, potenz in (("width", 1.0), ("height", 1.0), ("area", 2.0)):
        if sp not in g.columns:
            continue
        x = g[sp].to_numpy(dtype=float)
        roh = abstand(x)
        werte = []
        for lo, hi in ((0.93, 1.07), (0.75, 1.00)):
            w = [abstand(x * rng.uniform(lo, hi, x.size) ** potenz)
                 for _ in range(ziehungen)]
            werte.append(float(np.mean(w)))
        print(f"  {sp:<10}{roh:>10.4f}{werte[0]:>12.4f}{werte[1]:>12.4f}"
              f"{100 * werte[1] / roh:>11.0f} %")

    print()
    print("  Und was keine Skalierung je anfassen kann:")
    a = rank_auc(g["aspect"].to_numpy(dtype=float), ap)
    print(f"    Seitenverhaeltnis   AUC {a:.4f}   Abstand zur Muenze "
          f"{abs(a - 0.5):.4f}")
    print("    Skalierung ist isotrop, sie multipliziert Hoehe und Breite mit")
    print("    DERSELBEN Zahl. Das Seitenverhaeltnis bleibt unberuehrt.")

    print()
    print("  Welche Breite haette es gebraucht? Abstand auf 'width':")
    x = g["width"].to_numpy(dtype=float)
    roh = abstand(x)
    for lo, hi in ((0.75, 1.00), (0.75, 1.33), (0.70, 1.43), (0.60, 1.67)):
        w = [abstand(x * rng.uniform(lo, hi, x.size)) for _ in range(ziehungen)]
        m = float(np.mean(w))
        einseitig = " einseitig" if hi <= 1.0 or lo >= 1.0 else ""
        print(f"    {lo:.2f} bis {hi:.2f}  Faktor {hi / lo:.2f}x   "
              f"{m:.4f}   {100 * m / roh:>3.0f} % uebrig{einseitig}")


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ergebnisse", type=Path, default=Path("results_rsna.csv"))
    p.add_argument("--arm-dir", type=Path, default=Path(ARM_DIR))
    p.add_argument("--bezug-dir", type=Path, default=Path(BEZUG_DIR))
    p.add_argument("--geometrie", type=Path,
                   default=Path("predictions_rsna/crop_geometry.csv"))
    p.add_argument("--nur-urteil", action="store_true",
                   help="ohne den beschreibenden Teil")
    args = p.parse_args()

    if not args.ergebnisse.exists():
        abbruch(f"{args.ergebnisse} fehlt.")
    res = pd.read_csv(args.ergebnisse)

    block("HERKUNFT, vor jeder Zahl")
    r_arm = zeilen_holen(res, ARM_TAG)
    r_bez = zeilen_holen(res, BEZUG_TAG)
    if list(r_arm["fold"]) != list(r_bez["fold"]):
        abbruch(f"die Arme decken verschiedene Folds ab: "
                f"{list(r_arm['fold'])} gegen {list(r_bez['fold'])}. "
                f"Gepaart geht nur, was paarweise da ist.")
    folds = list(r_arm["fold"])
    print(f"  ok   beide Arme decken dieselben Folds: {folds}")

    staerke_pruefen(r_arm, ARM_TAG, SOLL_AUG, darf_leer=False)
    print(f"  ok   '{ARM_TAG}' trainierte mit Verschiebung "
          f"{SOLL_AUG['aug_translate']}, Skalierung "
          f"{SOLL_AUG['aug_scale_lo']} bis {SOLL_AUG['aug_scale_hi']}")
    staerke_pruefen(r_bez, BEZUG_TAG, ALT_AUG, darf_leer=True)
    print(f"  ok   '{BEZUG_TAG}' hat die alte Staerke oder stammt aus der "
          f"Zeit davor")

    n_arm = nachrechnen(args.arm_dir, folds)
    n_bez = nachrechnen(args.bezug_dir, folds)
    herkunft("Phase 6", r_arm, n_arm)
    herkunft("Phase 5", r_bez, n_bez)

    a_anker = float(n_bez["A"].mean())
    c_anker = float(n_bez["C"].mean())
    for name, ist, soll in (("A", a_anker, ANKER_A), ("C", c_anker, ANKER_C)):
        ab = abs(ist - soll)
        if ab > ANKER_TOLERANZ:
            abbruch(f"der Anker {name} rechnet sich zu {ist:.4f}, "
                    f"vorfestgelegt war {soll}. Der Bezugsarm ist nicht mehr "
                    f"der, gegen den vorfestgelegt wurde.")
        print(f"  ok   Anker {name} = {ist:.4f} wie vorfestgelegt ({soll})")

    block("DAS URTEIL, nach der Vorfestlegung vom 06.08.2026")
    print(f"  Anker aus {BEZUG_DIR} ueber {len(folds)} Folds: "
          f"A {a_anker:.4f}, C {c_anker:.4f}")
    print(f"  Phase 6:  A {n_arm['A'].mean():.4f}, C {n_arm['C'].mean():.4f}")

    rc = gepaart(n_arm["C"].to_numpy(), n_bez["C"].to_numpy())
    ra = gepaart(n_arm["A"].to_numpy(), n_bez["A"].to_numpy())
    sc = urteil_c(rc)
    sa = urteil_a(ra)
    zeige("PRIMAER   C, AUC(Score -> Projektion), muss FALLEN", rc, sc)
    zeige(f"NEBENBED  A, geschichtete AUC, Marge {MARGE_A}", ra, sa)

    block("PHASE 6 INSGESAMT")
    beide = sc[0] == "BESTANDEN" and sa[0] == "BESTANDEN"
    print(f"  primaer      C   {sc[0]}")
    print(f"  Nebenbedingung A {sa[0]}")
    print()
    if beide:
        print("  BESTANDEN. Die staerkere Augmentierung senkt den")
        print("  Projektionskanal, ohne die Trennschaerfe zu kosten. Damit")
        print("  waere Phase 7 in ihrer teuren Form moeglicherweise")
        print("  ueberfluessig.")
    elif sc[0] != "BESTANDEN" and sa[0] == "BESTANDEN":
        print("  DURCHGEFALLEN am primaeren Endpunkt. Die Augmentierung")
        print("  kostet nichts, sie leistet aber auch nicht das, wofuer sie")
        print("  angetreten ist. Phase 7 bleibt in ihrer teuren Form auf dem")
        print("  Tisch.")
        print()
        halbbreite = 0.5 * (rc["hi"] - rc["lo"])
        print("  'Nicht gefallen' heisst hier NICHT 'gesichert kein Effekt'.")
        print(f"  Das Intervall reicht von {rc['lo']:+.4f} bis "
              f"{rc['hi']:+.4f}, es haelt also einen")
        print("  Rueckgang genauso offen wie einen Anstieg. Diese Folds")
        print(f"  loesen auf C nur {halbbreite:.4f} auf, die Vorfestlegung")
        print("  hatte 0.017 erwartet.")
    else:
        print("  DURCHGEFALLEN an der Nebenbedingung. Bei gleichem Budget")
        print("  kostet diese Staerke Trennschaerfe. Der erlaubte")
        print("  Folgeversuch ist ein laengerer Arm bei derselben Staerke,")
        print("  mit eigener Vorfestlegung.")

    if not args.nur_urteil:
        stoerungen_zeigen(n_bez, n_arm)
        geometrie_simulieren(args.geometrie)
        block("WAS DER BESCHREIBENDE TEIL NICHT DARF")
        print("  Er darf das Urteil oben nicht umdeuten. Er sagt, WAS als")
        print("  Naechstes zu messen waere, nicht, wie dieser Lauf")
        print("  auszugehen hatte.")


if __name__ == "__main__":
    main()
