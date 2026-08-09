"""Phase 8: the verdict on 512 pixels, recomputed from scratch.

WHY THIS SCRIPT EXISTS AT ALL
-----------------------------
The training run already prints an AUC per fold. This script does not trust
those lines. It recomputes both endpoints from the per image prediction files,
compares the result against the numbers in `results_rsna.csv`, and refuses to
print a verdict if the two disagree. That is the standing rule of this project:
the conclusions of a phase get recomputed by a script that did not produce
them.

THE LEVER, AND HOW IT INVERTS THE PHASE 7 LOCKS
------------------------------------------------
Phase 7 changed `--images` and had to prove that the two arms saw DIFFERENT
pixels. Phase 8 changes `--size` and has to prove the opposite: the two arms
see the SAME files, and only the edge length differs. So the locks swap sides.

  1. `images` and `csv` must be IDENTICAL in both arms and must name
     data/rsna/png512 and data/rsna. In phase 7 a difference here was the
     point; here it is an abort.
  2. `size` must be 512 in the arm and 224 in the reference. This column did
     not exist before 08.08.2026; it was added for this phase, because
     `results_rsna.csv` had 76 columns and not one of them named the
     resolution. An arm that had silently run at 224 would have produced a
     clean looking null result.
  3. `input_px` must equal `size`. That one is not a self report: it is the
     edge length of the tensor the model actually received, measured on the
     first training batch. It catches the switch that is read and then not
     wired through, which has happened in this project before.
  4. `head_tile_coverage` must be UNCHANGED against phase 5. The coverage is
     an area fraction and barely depends on the pixel count; the expected
     values were computed on 08.08.2026 BEFORE the run, and the method was
     first checked against the phase 5 numbers, which it reproduces to 3.4e-05.
     This lock does NOT prove the resolution. It catches the other expensive
     mistake, an arm that ran on the crop, where the coverage sits near 0.18.
     Lock 3 is the resolution proof, not this one.

THE PRE-REGISTRATION, COPIED FROM erklaerungen/25_phase8_vorfestlegung.md
--------------------------------------------------------------------------
Written before the run. Anchors are the five fold means of the phase 5 winner
`predictions_final_model`, the same anchors phases 6 and 7 used:

    A = 0.8368    stratified AUC
    C = 0.7467    AUC(score -> ViewPosition)

TWO CO-PRIMARY ENDPOINTS. Phase 8 shows something if at least one gate opens.

GATE A, sharpness.  dA >= +0.008 AND the LOWER end of the paired 90 percent
    interval above zero. Both. The roadmap asked for 0.01; that was lowered to
    0.008 BEFORE the run and the lowering is marked as such in the document,
    because 0.01 lies above every A gain this project has ever measured from an
    image change (+0.0084, phase 7) and a gate that cannot open decides
    nothing.

GATE C, confounder.  The UPPER end of the paired interval below zero.

THE BOLT ON C. A passed gate C only counts if A is NON INFERIOR, lower end
    above -0.01. Otherwise the sentence is "a worse discriminator has less to
    give away", not "resolution clears the confounder".

MULTIPLICITY IS REPORTED, NOT CORRECTED. Two gates that can each carry the
    phase means roughly twice the chance that one opens by luck. Against pure
    noise stands that gate A also demands a minimum difference and that gate C
    tests a direction that was known in advance.

WHERE GATE C COMES FROM, AND WHY FOLD 0 IS REPORTED SEPARATELY
---------------------------------------------------------------
On 02.08.2026 a single fold ran at 320 px, built as a paired resolution test.
Only its A was ever reported, hence "320 px brings nothing". Recomputed on
08.08.2026 from the prediction files:

    A  0.8111 -> 0.8131   +0.0021  [-0.0065, +0.0105]
    C  0.7381 -> 0.7054   -0.0327  [-0.0388, -0.0267]

That is the second largest C effect ever measured in this project, behind the
full decoupling at -0.0554. It is one fold and one seed, the interval only
covers image sampling and not training runs, and the arm is an older
generation. A hypothesis with a direction, not a result.

It came out of FOLD 0, and fold 0 carries the same validation images here. So
the verdict runs over all five folds, and dC over folds 1 to 4 alone is printed
next to it. A large gap between the two means fold 0 is carrying the effect,
and that belongs in the result text. It is a disclosure, not a second chance.

USAGE
-----
    python rsna\\befunde\\rsna_phase8_auswertung.py
    python rsna\\befunde\\rsna_phase8_auswertung.py --nur-urteil
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
MARGE_A = 0.01           # non inferiority margin, the bolt on gate C
MINDEST_A = 0.008        # gate A, lowered from the roadmap's 0.01 BEFORE the run
NIVEAU = 0.90            # two sided, so the one sided bound is at 5 percent
GRAUZONE = -0.015        # point estimate at or below this: the follow up runs

# Die Aufloesung steht IM NAMEN des Arms. Eine zweite Kantenlaenge bekommt damit
# zwangslaeufig einen eigenen Tag und einen eigenen Vorhersageordner; teilten
# sie sich einen, ueberschrieben sie einander die Gewichte, und genau das ist in
# diesem Projekt schon einmal unbemerkt passiert.
ARM_TAG = "_p8s512"
ARM_DIR = "predictions_p8_s512"
BEZUG_TAG = "_p5head_ex"
BEZUG_DIR = "predictions_final_model"

ARM_SIZE = 512
BEZUG_SIZE = 224

# Beide Arme MUESSEN hierauf zeigen. In Phase 7 war der Unterschied dieser
# beiden Pfade der Hebel, hier ist er ein Abbruchgrund.
SOLL_IMAGES = "png512"
SOLL_CSV = "rsna"

# The augmentation both arms have to carry, and it is the OLD one. Phase 6
# failed; its strength is not the reference.
SOLL_AUG = dict(aug_translate=0.03, aug_scale_lo=0.93, aug_scale_hi=1.07,
                aug_degrees=7.0)

# The data side check, per fold, computed 08.08.2026 BEFORE the run.
# `head_tile_coverage` is an area fraction, so 512 px changes it only through
# the rounding of the box edges: the factor lies between 0.99989 and 1.00016.
# Method checked first against the phase 5 values, largest deviation 3.4e-05.
SOLL_COVERAGE = {0: 0.118765, 1: 0.117498, 2: 0.117710,
                 3: 0.116605, 4: 0.117179}
# Tight enough to tell this apart from the crop (0.18), loose enough for the
# rounding. NOT tight enough to say anything about the resolution; that is what
# `input_px` is for.
COVERAGE_TOLERANZ = 0.002

ANKER_TOLERANZ = 5e-5     # the anchors are quoted to four places
HERKUNFT_TOLERANZ = 1e-9  # recomputation against results_rsna.csv

# Configuration columns that must be IDENTICAL in both arms. `size` and
# `input_px` are deliberately NOT in here: they are the lever and they are
# checked separately, exactly as `head_tile_coverage` was in phase 7.
GLEICH = ["epochs", "seed", "cam_n", "balance_view", "balance_strength",
          "head", "head_grid", "head_negatives", "head_lambda_measured",
          "dml_index", "device_name", "n_fit", "n_sel", "n_val"]

STOERUNGEN = ["p_clean", "p_corners", "p_zoom_in", "p_shift", "p_rotate",
              "p_low_contr", "p_bright", "p_blur", "p_lowres"]
GEOMETRISCH = {"p_corners", "p_zoom_in", "p_shift"}

# Der 320-px-Lauf vom 02.08., neu gerechnet am 08.08. Steht hier, damit die
# Vorgeschichte im Bericht neben dem Ergebnis erscheint und nicht nur in einer
# Erinnerung. Ein Fold, ein Keim, aeltere Armgeneration.
PRIOR_320 = dict(fold=0, A_224=0.811071, A_320=0.813121,
                 C_224=0.738125, C_320=0.705445,
                 dA_lo=-0.0065, dA_hi=+0.0105,
                 dC_lo=-0.0388, dC_hi=-0.0267)


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
    alone.
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

def abbruch(text: str) -> None:
    print(f"\nABBRUCH: {text}")
    sys.exit(1)


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


def pfade_pruefen(d: pd.DataFrame, tag: str, darf_leer: bool = False) -> bool:
    """Schloss 1: beide Arme sehen DIESELBEN Dateien.

    In Phase 7 war ein Unterschied hier der Hebel. In Phase 8 ist er ein
    Abbruchgrund: unterscheiden sich die Bildordner, ist der Vergleich nicht
    mehr der zwischen zwei Aufloesungen, sondern der zwischen zwei Datensaetzen.

    WARUM `darf_leer` EXISTIERT, und warum das keine Aufweichung ist. Der
    Bezugsarm `_p5head_ex` lief am 04. bis 06.08.2026, also VOR den beiden
    Provenienzspalten, und hat sie deshalb leer. Fuer den ARM ist eine leere
    Spalte ein Abbruchgrund, denn dort ist sie vermeidbar. Fuer den Bezugsarm
    hiesse Abbruch, den Vergleichspartner der Phasen 6 und 7 neu rechnen zu
    muessen, und zwar wegen einer Spalte, deren Inhalt aus der Vorgabe von
    rsna_train.py folgt: `--images data/rsna/png512`, `--csv data/rsna`, und
    das sind genau die Pfade, die Phase 8 verlangt.

    Der zweite Beleg dafuer steht ohnehin weiter unten und ist der staerkere:
    rechneten sich die Anker A und C aus einem anderen Bildordner, waeren sie
    nicht mehr 0,8368 und 0,7467.

    Rueckgabe: True, wenn die Pfade wirklich dastanden, False bei leer.
    """
    for spalte, soll in (("images", SOLL_IMAGES), ("csv", SOLL_CSV)):
        if spalte not in d.columns:
            if darf_leer:
                print(f"  Hinweis: '{tag}' hat keine Spalte '{spalte}'. Der "
                      f"Bezugsarm ist aelter als der 07.08.2026; leer heisst "
                      f"die Vorgabe, und die Vorgabe ist das, was Phase 8 "
                      f"braucht. Belegt wird das ueber die Anker.")
                return False
            abbruch(f"results_rsna.csv hat keine Spalte '{spalte}'. Der Arm "
                    f"'{tag}' stammt aus einer Fassung von rsna_train.py vor "
                    f"dem 07.08.2026.")
        werte = [str(x) for x in d[spalte]]
        if any(w in ("", "nan") for w in werte):
            if darf_leer:
                print(f"  Hinweis: '{tag}' hat {spalte} leer. Der Bezugsarm "
                      f"lief vor der Provenienzspalte; leer heisst die "
                      f"Vorgabe (data/rsna/png512 bzw. data/rsna), und das "
                      f"ist genau, was Phase 8 verlangt. Belegt wird das "
                      f"ueber die Anker weiter unten.")
                return False
            abbruch(f"'{tag}' hat {spalte} leer. Dieser Fold wurde vor der "
                    f"Provenienzspalte gerechnet.")
        schlecht = [w for w in werte
                    if soll not in w.replace("/", "\\").split("\\")[-1]]
        if schlecht:
            abbruch(f"'{tag}' hat {spalte} = {sorted(set(schlecht))}, erwartet "
                    f"war ein Pfad, der auf '{soll}' endet. Phase 8 laeuft auf "
                    f"dem GANZEN Bild. Ein Arm auf dem Zuschnitt bewegte zwei "
                    f"Dinge auf einmal und beantwortete keines davon.")
    return True


def aufloesung_pruefen(d: pd.DataFrame, tag: str, soll: int,
                       darf_leer: bool) -> None:
    """Schloss 2 und 3: der Hebel selbst, einmal als Schalter und einmal
    gemessen.

    `size` ist die Selbstauskunft, `input_px` die Kantenlaenge des Tensors, den
    das Modell im ersten Stapel bekommen hat. Beide Spalten gibt es erst seit
    dem 08.08.2026, und genau das ist der Grund, warum es sie gibt: bis dahin
    stand die Aufloesung in keiner der 76 Spalten.
    """
    if "size" not in d.columns or "input_px" not in d.columns:
        if darf_leer:
            print(f"  Hinweis: '{tag}' hat keine Spalten size/input_px. Der "
                  f"Bezugsarm stammt aus der Zeit davor; seine Aufloesung "
                  f"belegt statt dessen die Ankerpruefung weiter unten.")
            return
        abbruch("results_rsna.csv hat keine Spalten size/input_px. Der Arm "
                f"'{tag}' stammt aus einer Fassung von rsna_train.py vor dem "
                f"08.08.2026, und dann ist der einzige Hebel dieser Phase "
                f"nirgends belegt. Ein Arm, der versehentlich bei 224 px "
                f"gelaufen waere, saehe wie ein sauberes Nullergebnis aus.")
    s = pd.to_numeric(d["size"], errors="coerce").to_numpy(dtype=float)
    i = pd.to_numeric(d["input_px"], errors="coerce").to_numpy(dtype=float)
    if np.all(np.isnan(s)):
        if darf_leer:
            print(f"  Hinweis: '{tag}' hat size leer. Leer heisst 224, siehe "
                  f"rsna_train.py; belegt ist das erst ueber die Anker.")
            return
        abbruch(f"'{tag}' hat size leer.")
    if not np.allclose(s, soll, equal_nan=False):
        abbruch(f"'{tag}' hat size = {sorted(set(s))}, erwartet {soll}. "
                f"DIESER ARM LIEF NICHT BEI {soll} PIXELN.")
    if np.all(np.isnan(i)):
        abbruch(f"'{tag}' hat input_px leer, obwohl size gesetzt ist. Dann "
                f"belegt nur die Selbstauskunft die Aufloesung.")
    if not np.allclose(i, s, equal_nan=False):
        abbruch(f"'{tag}': size {sorted(set(s))} gegen gemessenes input_px "
                f"{sorted(set(i))}. Der Schalter ist nicht dort angekommen, "
                f"wo er wirkt.")
    print(f"  ok   '{tag}' lief bei {soll} px, Schalter und gemessene "
          f"Kantenlaenge stimmen ueberein")


def ein_hebel_pruefen(arm: pd.DataFrame, bez: pd.DataFrame,
                      pfade_vergleichbar: bool = True) -> None:
    """Schloss 4: unterscheidet sich der Arm WIRKLICH nur in der Aufloesung?

    Diese Pruefung schreibt nichts vor, sie liest nach. Sie faengt den Fall, in
    dem beim Zusammenbauen der Befehlszeile ausser der Kantenlaenge noch etwas
    anderes verrutscht ist, etwa der Adapter oder die Zahl der Epochen. Solche
    Zweitunterschiede sind hinterher nicht mehr auffindbar.
    """
    fehler = []
    for spalte in GLEICH:
        if spalte not in arm.columns or spalte not in bez.columns:
            continue
        a, b = arm[spalte].to_numpy(), bez[spalte].to_numpy()
        try:
            gleich = np.allclose(a.astype(float), b.astype(float),
                                 rtol=0, atol=1e-12, equal_nan=True)
        except (TypeError, ValueError):
            gleich = list(map(str, a)) == list(map(str, b))
        if not gleich:
            fehler.append(f"{spalte}: Arm {sorted(set(map(str, a)))} gegen "
                          f"Bezug {sorted(set(map(str, b)))}")
    if fehler:
        abbruch("die beiden Arme unterscheiden sich in mehr als in der "
                "Aufloesung:\n    " + "\n    ".join(fehler) +
                "\n  Ein gepaarter Vergleich mit zwei Unterschieden "
                "beantwortet keine Frage.")
    print(f"  ok   beide Arme stimmen in {len(GLEICH)} Konfigurationsspalten "
          f"ueberein")
    # Und die Pfade muessen zwischen den Armen GLEICH sein, nicht nur je Arm
    # richtig. Das ist die Umkehrung von Phase 7 und deshalb leicht zu
    # vergessen.
    #
    # Uebersprungen, wenn der Bezugsarm die Spalten leer hat: dann steht dort
    # nichts, was sich vergleichen liesse, und ein Vergleich gegen die leere
    # Zeichenkette braeche jeden richtigen Lauf ab. Siehe pfade_pruefen.
    if not pfade_vergleichbar:
        print("  --   Pfadvergleich zwischen den Armen uebersprungen, der "
              "Bezugsarm hat die Spalten leer")
        return
    for spalte in ("images", "csv"):
        if spalte in arm.columns and spalte in bez.columns:
            a = {str(x).replace("/", "\\").rstrip("\\") for x in arm[spalte]}
            b = {str(x).replace("/", "\\").rstrip("\\") for x in bez[spalte]}
            if a != b:
                abbruch(f"{spalte}: Arm {sorted(a)} gegen Bezug {sorted(b)}. "
                        f"Phase 8 aendert die Aufloesung, nicht die Dateien.")
    print("  ok   beide Arme zeigen auf dieselben Bild- und Kastenordner")


def staerke_pruefen(d: pd.DataFrame, tag: str, soll: dict,
                    darf_leer: bool) -> None:
    """Die Augmentierungsstaerke, geprueft statt angenommen.

    Es steht die ALTE Staerke hier, wie in Phase 7. Phase 6 ist durchgefallen,
    ihr Arm ist nicht der Bezug. Ein Phase-8-Arm, der die Phase-6-Staerke
    mitgeschleppt haette, bewegte zwei Dinge gleichzeitig.
    """
    for spalte, wert in soll.items():
        if spalte not in d.columns:
            if darf_leer:
                continue
            abbruch(f"'{tag}' hat keine Spalte {spalte}.")
        ist = pd.to_numeric(d[spalte], errors="coerce").to_numpy(dtype=float)
        if np.all(np.isnan(ist)):
            if darf_leer:
                continue
            abbruch(f"'{tag}' hat {spalte} leer. Die Staerke ist nicht belegt.")
        if not np.allclose(ist, wert, atol=1e-9, equal_nan=False):
            abbruch(f"'{tag}' hat {spalte} = {sorted(set(ist))}, erwartet "
                    f"{wert}. Phase 8 laeuft mit der ALTEN "
                    f"Augmentierungsstaerke.")


def abdeckung_pruefen(d: pd.DataFrame, bez: pd.DataFrame) -> None:
    """Schloss 5: der Arm lief auf dem GANZEN Bild, nicht auf dem Zuschnitt.

    `head_tile_coverage` ist der Anteil der Bildflaeche, den die Kaesten der
    Fit-Bilder bedecken, gerechnet aus den Kaesten in --csv. Als Flaechenanteil
    haengt er fast nicht an der Kantenlaenge: 512 px aendert ihn nur ueber die
    Rundung der Kastenraender, Faktor 0,99989 bis 1,00016.

    WAS DIESE ZAHL NICHT KANN: sie belegt die Aufloesung NICHT. Sie kann 224
    und 512 nicht unterscheiden, und das ist hier ausdruecklich gesagt, damit
    sie niemand fuer den Beleg haelt. Sie faengt den anderen teuren Fehler:
    liefe der Arm auf crop512_fix080, laege sie bei rund 0,18.
    """
    if "head_tile_coverage" not in d.columns:
        abbruch("results_rsna.csv hat keine Spalte head_tile_coverage.")
    print(f"\n  {'Fold':>5}{'Phase 5':>10}{'Phase 8 ist':>14}"
          f"{'soll':>10}{'Faktor':>9}")
    for _, r in d.iterrows():
        f = int(r["fold"])
        ist = float(r["head_tile_coverage"])
        soll = SOLL_COVERAGE.get(f)
        alt = float(bez[bez["fold"] == f]["head_tile_coverage"].iloc[0])
        if soll is None:
            print(f"  {f:>5}{alt:>10.4f}{ist:>14.4f}{'-':>10}"
                  f"{ist / alt:>9.4f}   (kein Sollwert vorfestgelegt)")
            continue
        print(f"  {f:>5}{alt:>10.4f}{ist:>14.4f}{soll:>10.4f}"
              f"{ist / alt:>9.4f}")
        if abs(ist - soll) > COVERAGE_TOLERANZ:
            abbruch(
                f"Fold {f}: die Kastenabdeckung steht bei {ist:.4f}, "
                f"vorfestgelegt war {soll:.4f} (Toleranz "
                f"{COVERAGE_TOLERANZ}).\n"
                f"  Bei rund 0,18 laege sie, wenn der Lauf auf dem ZUSCHNITT "
                f"gelaufen waere.\n"
                f"  Die Sollwerte stehen in erklaerungen/"
                f"25_phase8_vorfestlegung.md.")
    print("  ok   die Kastenabdeckung ist unveraendert. Der Arm lief auf dem")
    print("       GANZEN Bild. (Ueber die Aufloesung sagt diese Zahl nichts,")
    print("       das belegt input_px.)")


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

def urteil_a(r: dict) -> tuple[str, str]:
    """Gate A: A must RISE, by at least MINDEST_A, interval clear of zero."""
    if r["lo"] > 0 and r["mean"] >= MINDEST_A:
        return "BESTANDEN", (f"gesichert gestiegen und der Punktwert liegt "
                             f"auf oder ueber {MINDEST_A:+.3f}")
    if r["lo"] > 0:
        return "DURCHGEFALLEN", (f"gesichert gestiegen, aber nur um "
                                 f"{r['mean']:+.4f}, also unter dem "
                                 f"Mindestunterschied {MINDEST_A:+.3f}")
    if r["hi"] < 0:
        return "DURCHGEFALLEN", "A ist gesichert GEFALLEN"
    return "DURCHGEFALLEN", "das Intervall enthaelt die Null"


def urteil_a_nichtunterlegen(r: dict) -> tuple[str, str]:
    """The bolt on gate C. The LOWER end decides."""
    if r["lo"] > -MARGE_A:
        return "BESTANDEN", (f"das untere Ende liegt ueber {-MARGE_A:+.2f}, "
                             f"A ist nicht unterlegen")
    return "DURCHGEFALLEN", (f"das untere Ende liegt unter {-MARGE_A:+.2f}, "
                             f"ein Verlust ueber der Marge ist moeglich")


def urteil_c(r: dict) -> tuple[str, str]:
    """Gate C: C must FALL. A rise is the problem, so the UPPER end decides."""
    if r["hi"] < 0:
        return "BESTANDEN", "das obere Ende liegt unter null, C faellt"
    if r["lo"] > 0:
        return "DURCHGEFALLEN", "C ist gesichert GESTIEGEN"
    if r["mean"] <= GRAUZONE:
        return "GRAUZONE", (f"das Intervall enthaelt die Null, der Punktwert "
                            f"liegt aber bei {r['mean']:+.4f}, also auf oder "
                            f"unter {GRAUZONE:+.3f}")
    return "DURCHGEFALLEN", ("das obere Ende liegt ueber null und der "
                             "Punktwert ueber der Grauzonenschwelle")


def block(titel: str) -> None:
    print("\n" + "=" * 78)
    print(titel)
    print("=" * 78)


def zeige(name: str, r: dict, spruch: tuple[str, str],
          erwartet: float | None = None) -> None:
    print(f"\n  {name}")
    print("    je Fold  " + "  ".join(f"{x:+.4f}" for x in r["je_fold"]))
    print(f"    Differenz {r['mean']:+.4f} +- {r['sd']:.4f}   "
          f"{int(NIVEAU * 100)}-Prozent-Intervall "
          f"[{r['lo']:+.4f}, {r['hi']:+.4f}]")
    hinweis = ("" if erwartet is None
               else f"   (vorher erwartet: rund {erwartet:.3f})")
    print(f"    Halbbreite {0.5 * (r['hi'] - r['lo']):.4f}{hinweis}")
    print(f"    {spruch[0]}: {spruch[1]}")


def fold0_offenlegen(n_arm: pd.DataFrame, n_bez: pd.DataFrame,
                     folds: list) -> None:
    """Die Ehrlichkeitsklausel aus der Vorfestlegung, Abschnitt 'Fold 0'.

    Die C-Vermutung stammt aus Fold 0 des 320-px-Laufs, und Fold 0 hat hier
    dieselben Bewertungsbilder. Also wird dC ohne ihn danebengestellt. Das ist
    keine zweite Chance: das Urteil oben steht und wird davon nicht angefasst.
    """
    if 0 not in folds or len(folds) < 3:
        print("  Fold 0 ist nicht dabei oder es sind zu wenige Folds, "
              "uebersprungen.")
        return
    m = n_arm["fold"] != 0
    ohne = gepaart(n_arm.loc[m, "C"].to_numpy(), n_bez.loc[m, "C"].to_numpy())
    mit = gepaart(n_arm["C"].to_numpy(), n_bez["C"].to_numpy())
    f0 = float(n_arm.loc[~m, "C"].iloc[0] - n_bez.loc[~m, "C"].iloc[0])
    print(f"  dC ueber alle {mit['n']} Folds        {mit['mean']:+.4f}  "
          f"[{mit['lo']:+.4f}, {mit['hi']:+.4f}]")
    print(f"  dC ohne Fold 0 ({ohne['n']} Folds)     {ohne['mean']:+.4f}  "
          f"[{ohne['lo']:+.4f}, {ohne['hi']:+.4f}]")
    print(f"  dC auf Fold 0 allein          {f0:+.4f}")
    print()
    unterschied = abs(mit["mean"] - ohne["mean"])
    if unterschied > 0.010:
        print(f"  ACHTUNG: die beiden Punktwerte liegen {unterschied:.4f} "
              f"auseinander.")
        print("  Fold 0 traegt den Effekt mit, und Fold 0 ist der Fold, aus")
        print("  dem die Vermutung stammt. Das gehoert in den Ergebnistext.")
    else:
        print(f"  Die beiden Punktwerte liegen {unterschied:.4f} auseinander, "
              f"also")
        print("  traegt Fold 0 den Effekt nicht allein.")


# --------------------------------------------------------------------------
# Descriptive part
# --------------------------------------------------------------------------

def stoerungen_zeigen(bez: pd.DataFrame, arm: pd.DataFrame) -> None:
    block("BESCHREIBEND 1: C unter Stoerung des Testbildes")
    print("  Nicht vorfestgelegt. Die Frage ist, WORIN der Projektionshinweis")
    print("  steckt. Phase 6 und 7 fanden beide: keine geometrische Stoerung")
    print("  senkt C, nur die Helligkeit. Wenn mehr Aufloesung wirkt, weil sie")
    print("  dem globalen Grauwert Textur an die Seite stellt, muesste der")
    print("  Helligkeitsausschlag hier KLEINER werden als in Phase 5.")
    print()
    print(f"  {'Stoerung':<12}{'Phase 5':>10}{'Phase 8':>10}"
          f"{'P5 gg clean':>13}{'P8 gg clean':>13}")
    b0 = bez["C_clean"].mean()
    a0 = arm["C_clean"].mean()
    for s in STOERUNGEN:
        k = "C_" + s[2:]
        b, a = bez[k].mean(), arm[k].mean()
        mark = "  <-- Rahmung" if s in GEOMETRISCH else ""
        if s == "p_bright":
            mark = "  <-- der Kanal"
        print(f"  {s[2:]:<12}{b:>10.4f}{a:>10.4f}{b - b0:>+13.4f}"
              f"{a - a0:>+13.4f}{mark}")


def prior_zeigen() -> None:
    block("BESCHREIBEND 2: der 320-px-Lauf vom 02.08., neu gerechnet")
    p = PRIOR_320
    print("  Ein Fold, ein Keim, aeltere Armgeneration (bal10, ohne Kopf,")
    print("  Adapter 0). Gebaut als gepaarter Aufloesungsversuch: gleiche")
    print("  Variante, gleicher Keim, acht Epochen, nur die Kantenlaenge")
    print("  anders. Berichtet wurde davon bis zum 08.08. nur die A-Zeile.")
    print()
    print(f"  {'':<6}{'224 px':>10}{'320 px':>10}{'Differenz':>12}"
          f"{'Intervall ueber die Bilder':>30}")
    for name in ("A", "C"):
        alt, neu = p[f"{name}_224"], p[f"{name}_320"]
        spanne = "[%+.4f, %+.4f]" % (p[f"d{name}_lo"], p[f"d{name}_hi"])
        print(f"  {name:<6}{alt:>10.4f}{neu:>10.4f}{neu - alt:>+12.4f}"
              f"{spanne:>30}")
    print()
    print("  Das Intervall stammt aus einem Bootstrap ueber die BILDER und")
    print("  misst nicht die Streuung zwischen Trainingslaeufen. Es ist zu eng.")


def vergleich_zeigen(res: pd.DataFrame, n_bez: pd.DataFrame,
                     n_arm: pd.DataFrame) -> None:
    block("BESCHREIBEND 3: alle Arme am selben Anker")
    print("  Alle haengen an predictions_final_model, sind also untereinander")
    print("  lesbar. Beschreibend; die Urteile stehen in ihren eigenen Phasen.")
    print()
    print(f"  {'Arm':<28}{'A':>9}{'dA':>10}{'C':>9}{'dC':>10}")
    print(f"  {'Bezug (Phase 5)':<28}{n_bez['A'].mean():>9.4f}{'':>10}"
          f"{n_bez['C'].mean():>9.4f}")
    for tag, name in (("_p6aug", "Phase 6, Augmentierung"),
                      ("_p7fix080", "Phase 7, fester Zuschnitt")):
        d = res[res["tag"] == tag]
        if d.empty:
            continue
        d = d.drop_duplicates(subset="fold", keep="last").sort_values("fold")
        if list(d["fold"]) != list(n_bez["fold"]):
            print(f"  {name:<28}(andere Folds, kein gepaarter Vergleich)")
            continue
        a, c = float(d["auc_stratified"].mean()), float(d["auc_view"].mean())
        print(f"  {name:<28}{a:>9.4f}{a - n_bez['A'].mean():>+10.4f}"
              f"{c:>9.4f}{c - n_bez['C'].mean():>+10.4f}")
    a, c = float(n_arm["A"].mean()), float(n_arm["C"].mean())
    print(f"  {'Phase 8, 512 px':<28}{a:>9.4f}{a - n_bez['A'].mean():>+10.4f}"
          f"{c:>9.4f}{c - n_bez['C'].mean():>+10.4f}")


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ergebnisse", type=Path, default=Path("results_rsna.csv"))
    p.add_argument("--arm-dir", type=Path, default=Path(ARM_DIR))
    p.add_argument("--bezug-dir", type=Path, default=Path(BEZUG_DIR))
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

    # --- die Schloesser um den einen Hebel --------------------------------
    pfade_pruefen(r_arm, ARM_TAG)
    print(f"  ok   '{ARM_TAG}' trainierte auf "
          f"{sorted(set(r_arm['images']))[0]}")
    bez_pfade_da = pfade_pruefen(r_bez, BEZUG_TAG, darf_leer=True)
    aufloesung_pruefen(r_arm, ARM_TAG, ARM_SIZE, darf_leer=False)
    aufloesung_pruefen(r_bez, BEZUG_TAG, BEZUG_SIZE, darf_leer=True)
    ein_hebel_pruefen(r_arm, r_bez, pfade_vergleichbar=bez_pfade_da)
    staerke_pruefen(r_arm, ARM_TAG, SOLL_AUG, darf_leer=False)
    staerke_pruefen(r_bez, BEZUG_TAG, SOLL_AUG, darf_leer=True)
    print(f"  ok   '{ARM_TAG}' hat die ALTE Augmentierung (Verschiebung "
          f"{SOLL_AUG['aug_translate']}, Skalierung "
          f"{SOLL_AUG['aug_scale_lo']} bis {SOLL_AUG['aug_scale_hi']})")
    print(f"  ok   '{BEZUG_TAG}' hat dieselbe Staerke oder stammt aus der "
          f"Zeit vor den Schaltern")
    abdeckung_pruefen(r_arm, r_bez)

    block("DIE ENDPUNKTE, aus den Vorhersagen je Bild nachgerechnet")
    n_arm = nachrechnen(args.arm_dir, folds)
    n_bez = nachrechnen(args.bezug_dir, folds)
    herkunft("Phase 8", r_arm, n_arm)
    herkunft("Phase 5", r_bez, n_bez)

    a_anker = float(n_bez["A"].mean())
    c_anker = float(n_bez["C"].mean())
    for name, ist, soll in (("A", a_anker, ANKER_A), ("C", c_anker, ANKER_C)):
        if abs(ist - soll) > ANKER_TOLERANZ:
            abbruch(f"der Anker {name} rechnet sich zu {ist:.4f}, "
                    f"vorfestgelegt war {soll}. Der Bezugsarm ist nicht mehr "
                    f"der, gegen den vorfestgelegt wurde.")
        print(f"  ok   Anker {name} = {ist:.4f} wie vorfestgelegt ({soll})")

    block("DAS URTEIL, nach der Vorfestlegung vom 08.08.2026")
    print(f"  Anker aus {BEZUG_DIR} ueber {len(folds)} Folds: "
          f"A {a_anker:.4f}, C {c_anker:.4f}")
    print(f"  Phase 8:  A {n_arm['A'].mean():.4f}, C {n_arm['C'].mean():.4f}")
    print()
    print("  A misst, ob der Score die PNEUMONIE vorhersagt, und soll steigen.")
    print("  C misst, ob derselbe Score die AUFNAHMEART vorhersagt, und soll")
    print("  fallen. Zwei gleichrangige Tore; die Phase zeigt etwas, wenn")
    print("  mindestens eines aufgeht.")

    ra = gepaart(n_arm["A"].to_numpy(), n_bez["A"].to_numpy())
    rc = gepaart(n_arm["C"].to_numpy(), n_bez["C"].to_numpy())
    sa = urteil_a(ra)
    sn = urteil_a_nichtunterlegen(ra)
    sc = urteil_c(rc)
    zeige(f"TOR A   geschichtete AUC, muss um mindestens {MINDEST_A:+.3f} "
          f"STEIGEN", ra, sa)
    zeige("TOR C   AUC(Score -> Projektion), muss FALLEN", rc, sc,
          erwartet=0.025)
    print(f"\n  RIEGEL auf Tor C: A nicht unterlegen, Marge {MARGE_A}")
    print(f"    {sn[0]}: {sn[1]}")

    block("DIE EHRLICHKEITSKLAUSEL: dC ohne Fold 0")
    fold0_offenlegen(n_arm, n_bez, folds)

    block("PHASE 8 INSGESAMT")
    c_zaehlt = sc[0] == "BESTANDEN" and sn[0] == "BESTANDEN"
    print(f"  Tor A   {sa[0]}")
    print(f"  Tor C   {sc[0]}" +
          ("" if sn[0] == "BESTANDEN" else "   (Riegel offen, zaehlt nicht)"))
    print()
    if sa[0] == "BESTANDEN" and c_zaehlt:
        print("  BESTANDEN AN BEIDEN TOREN. 512 px bringt Trennschaerfe UND")
        print("  senkt den Projektionskanal. Das ist der erste Eingriff seit")
        print("  der vollen Entkopplung, der beides tut. 512 px ist damit")
        print("  Kandidat fuer das Endrezept, und Phase 10 muss klaeren, was")
        print("  fuenf ResNet18 bei 512 px auf der CPU der Webapp kosten.")
    elif sa[0] == "BESTANDEN":
        print("  BESTANDEN AN TOR A. Mehr Aufloesung bringt Trennschaerfe.")
        print("  Die Aufloesungsachse ist damit zum ersten Mal offen; sie war")
        print("  am 26.07. als Vorhersage und am 02.08. bei 320 px gescheitert.")
        print("  Tor C ist nicht aufgegangen, der 320-px-Hinweis hat sich also")
        print("  nicht bestaetigt.")
    elif c_zaehlt:
        print("  BESTANDEN AN TOR C. Mehr Aufloesung senkt den")
        print("  Projektionskanal, ohne Trennschaerfe zu kosten. Der")
        print("  320-px-Hinweis hat sich bestaetigt, und die Erklaerung dazu")
        print("  steht in der Stoerungstabelle unten: der Kanal sitzt im")
        print("  globalen Grauwert, und feine Textur stellt sich ihm an die")
        print("  Seite. Der photometrische Arm greift dasselbe direkt an und")
        print("  wird damit zur Bestaetigung mit anderem Mittel.")
    elif sc[0] == "GRAUZONE" and sn[0] == "BESTANDEN":
        print("  GRAUZONE auf Tor C. Das Intervall enthaelt die Null, der")
        print(f"  Punktwert liegt bei {rc['mean']:+.4f}, also auf oder unter")
        print(f"  der vorher festgelegten Schwelle {GRAUZONE:+.3f}.")
        print()
        print("  DER VORFESTGELEGTE FOLGEVERSUCH: drei Keime je Fold in BEIDEN")
        print("  Armen, 15 gepaarte Einheiten, Halbbreite auf C dann rund")
        print("  0,013. Bei 512 px kostet er rund 33 Stunden. Das ist eine")
        print("  Entscheidung ueber Rechenzeit und keine ueber Statistik: die")
        print("  Schwelle stand vorher, der Preis auch.")
    elif sn[0] != "BESTANDEN":
        print("  DURCHGEFALLEN, und der Riegel hat gegriffen. A ist um mehr")
        print("  als die Marge gefallen. Der Satz lautet: BEI ACHT EPOCHEN")
        print("  kostet 512 px Trennschaerfe. Ein C-Rueckgang waere hier nicht")
        print("  zu verwerten, denn ein schlechterer Trenner hat auch weniger")
        print("  zu verraten. Erlaubter Folgeversuch: mehr Epochen bei 512 px,")
        print("  mit eigener Vorfestlegung, denn die vierfache Bildflaeche bei")
        print("  gleicher Epochenzahl ist nicht dasselbe Trainingsbudget.")
    elif rc["lo"] > 0:
        print("  DURCHGEFALLEN an beiden Toren, und C ist gesichert GESTIEGEN.")
        print("  Die Aufloesungsachse ist gegen den Confounder geschlossen,")
        print("  der 320-px-Hinweis war ein Foldeffekt. Uebrig bleibt der")
        print("  photometrische Arm.")
    else:
        print("  DURCHGEFALLEN an beiden Toren.")
        print()
        print("  Auf A ist das der dritte Fehlschlag der Aufloesungsachse nach")
        print("  dem 26.07. und dem 02.08. Damit ist sie zu, und Phase 9")
        print("  entfaellt endgueltig; sie haette ohnehin nur die Rahmung")
        print("  beizutragen, denn der Zuschnitt traegt bei 512 px nur 410")
        print("  echte Bildpunkte.")
        print()
        print("  Auf C hat sich der 320-px-Hinweis nicht bestaetigt. Das ist")
        print("  ein Ergebnis ueber ihn und kein fehlendes: die Phase war fuer")
        print("  die volle vermutete Wirkung genau genug.")
        print()
        print("  Der naechste Arm ist der photometrische.")

    # Nur wenn die Null WIRKLICH im Intervall liegt.
    if sc[0] != "BESTANDEN" and rc["lo"] <= 0 <= rc["hi"]:
        halb = 0.5 * (rc["hi"] - rc["lo"])
        print()
        print("  'Nicht gefallen' heisst NICHT 'gesichert kein Effekt'. Das")
        print(f"  Intervall auf C reicht von {rc['lo']:+.4f} bis {rc['hi']:+.4f};")
        print(f"  diese Folds loesen {halb:.4f} auf, die Vorfestlegung hatte")
        print("  rund 0,025 erwartet.")

    print()
    print("  ZUR VIELFACHHEIT: zwei Tore, an denen die Phase bestehen kann,")
    print("  heisst rund die doppelte Wahrscheinlichkeit, dass eines zufaellig")
    print("  aufgeht. Das wird nicht korrigiert, sondern hier gesagt.")

    if not args.nur_urteil:
        stoerungen_zeigen(n_bez, n_arm)
        prior_zeigen()
        vergleich_zeigen(res, n_bez, n_arm)
        block("WAS DER BESCHREIBENDE TEIL NICHT DARF")
        print("  Er darf das Urteil oben nicht umdeuten. Er sagt, WAS als")
        print("  Naechstes zu messen waere, nicht, wie dieser Lauf")
        print("  auszugehen hatte.")


if __name__ == "__main__":
    main()
