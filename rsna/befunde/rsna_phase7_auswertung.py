"""Phase 7: the verdict on the fixed crop, recomputed from scratch.

WHY THIS SCRIPT EXISTS AT ALL
-----------------------------
The training run already prints an AUC per fold. This script does not trust
those lines. It recomputes both endpoints from the per image prediction files,
compares the result against the numbers in `results_rsna.csv`, and refuses to
print a verdict if the two disagree. That is the standing rule of this project:
the conclusions of a phase get recomputed by a script that did not produce
them.

WHAT IT REFUSES TO DO, AND WHY THAT IS THE POINT OF PHASE 7
-----------------------------------------------------------
Phase 7 has exactly ONE lever, and until 07.08.2026 that lever appeared nowhere
in the output: neither `--images` nor `--csv` were logged or stored. An arm
that had accidentally trained on `png512` would have produced a clean looking
null result and nothing would have contradicted it. Phase 6 built
`staerke_pruefen` against exactly this failure; here it needs three locks
instead of one, because the lever is a path rather than a number:

  1. `images` and `csv` in `results_rsna.csv` must name the crop folder. That
     is a self report: it says what the run believed it was doing.
  2. The arm must differ from the reference arm in NOTHING ELSE. Every other
     configuration column is compared, and a second difference aborts.
  3. `head_tile_coverage` must have RISEN by the factor the crop implies. This
     one is not a self report: it is computed from the boxes the run actually
     loaded, and the fixed crop magnifies every box by 1/0.80 linearly. The
     expected value per fold was computed on 07.08.2026, BEFORE the run, and
     the method was first checked against the phase 5 numbers, which it
     reproduces to 8e-17. See erklaerungen/23_phase7_zuschnitt.md.

THE PRE-REGISTRATION, COPIED FROM erklaerungen/23_phase7_zuschnitt.md
---------------------------------------------------------------------
Written before the run. Anchors are the five fold means of the phase 5 winner
`predictions_final_model`, the same anchors phase 6 used:

    A = 0.8368    stratified AUC
    C = 0.7467    AUC(score -> ViewPosition)

PRIMARY, C FALLS. Paired per fold against the anchor, 90 percent interval of
    the difference, the UPPER end must lie below zero.

CONSTRAINT, A IS NON INFERIOR. Margin 0.01, paired per fold, 90 percent
    interval, the LOWER end must lie above -0.01.

BOTH must hold. The roadmap gate "C must fall from 0.8166" is not used: 0.8166
is the phase 0 baseline WITHOUT reweighting, every arm since phase 5 starts
below it, and a gate that saturates decides nothing.

THE READING RULE, ALSO PRE-REGISTERED
--------------------------------------
The paired comparison resolves about 0.025 on C, and the honest expectation
written down before the run was that phase 7 might well land underneath that.
So the grey zone has its own branch, with its own threshold, fixed in advance:

    upper end below zero                  -> PASSED
    interval covers zero, point <= -0.010 -> GREY ZONE, the pre-registered
                                             follow up is three seeds per fold
    interval covers zero, point >  -0.010 -> FAILED, the FRAMING is exonerated
    lower end above zero                  -> C ROSE, the crop axis is closed
    A fails                               -> the sentence is about eight
                                             epochs, not about crops

"THE FRAMING", NOT "GEOMETRY", AND THE DIFFERENCE COST A MEASUREMENT
--------------------------------------------------------------------
That third branch first read "geometry is exonerated". It was corrected on the
evening of 07.08.2026, after the crop had been produced but before any arm was
trained, and the correction only ever makes the conclusion weaker.

`qc/crop_varianten_tabelle.csv` measures the geometry OF THE CROP WINDOW:
0.7144 adaptive against 0.5545 fixed, so three quarters of that channel go
away. The model never sees window parameters. It sees pixels, and those still
contain a lung. Measured in the same instrument (`rsna_restkanal.py`, first
calibrated until it reproduces both table rows), the lung's own rectangle
carries 0.2610 in the whole image and still 0.1638 inside the crop: the crop
takes 37 percent of what the model can see, not 75.

The remainder is anatomy rather than framing, and it cannot be cropped away
without cropping away the finding, because an opacity IS partly a change of
size and shape. So a null result on C says the framing did not carry the
channel. It does not say geometry did not.

No anchor, gate, margin or threshold was touched by this.

USAGE
-----
    python rsna\\befunde\\rsna_phase7_auswertung.py
    python rsna\\befunde\\rsna_phase7_auswertung.py --nur-urteil
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
GRAUZONE = -0.010        # point estimate at or below this: the follow up runs

# Der Zuschnitt steht IM NAMEN des Arms, nicht nur im Ordner nebenan. Eine
# zweite Seitenlaenge bekommt damit zwangslaeufig einen eigenen Tag und einen
# eigenen Vorhersageordner; teilten sie sich einen, ueberschrieben sie
# einander die Gewichte, und genau das ist in diesem Projekt schon einmal
# unbemerkt passiert.
ARM_TAG = "_p7fix080"
ARM_DIR = "predictions_p7_fix080"
BEZUG_TAG = "_p5head_ex"
BEZUG_DIR = "predictions_final_model"

# The crop the arm claims to have trained on.
CROP_ORDNER = "crop512_fix080"
FESTE_SEITE = 0.80
VERSATZ_Y = 0.03

# The augmentation the arm has to carry, and it is the OLD one. Phase 6 failed;
# its strength is not the reference. An arm that dragged the phase 6 numbers
# along would move two things at once and could not answer anything.
SOLL_AUG = dict(aug_translate=0.03, aug_scale_lo=0.93, aug_scale_hi=1.07,
                aug_degrees=7.0)

# The data side proof of the crop, per fold, computed 07.08.2026 BEFORE the
# run. `head_tile_coverage` comes from the boxes in --csv; the fixed crop
# magnifies every box by 1/0.80 linearly, so it has to rise by about 1.554.
# Method checked first against the phase 5 values, reproduced to 8e-17.
SOLL_COVERAGE = {0: 0.184595, 1: 0.182570, 2: 0.182977,
                 3: 0.181224, 4: 0.182100}
# Loose enough for rounding, tight enough to tell the crop variants apart: a
# fixed side of 0.85 would land near 0.164, ten times this tolerance away.
COVERAGE_TOLERANZ = 0.002

ANKER_TOLERANZ = 5e-5     # the anchors are quoted to four places
HERKUNFT_TOLERANZ = 1e-9  # recomputation against results_rsna.csv

# Configuration columns that must be IDENTICAL in both arms. The augmentation
# columns are handled separately, because the reference arm predates the
# switches and legitimately has them empty.
#
# Only what is SET goes in here, never what comes OUT. `head_lambda` is the
# effective lambda and it is MEASURED on the first usable batch, so it differs
# between two arms that are configured identically; putting it here would abort
# every correct run. `head_lambda_measured` is the flag that says the
# measurement happened, and that is a setting. Same line runs between
# `head_grid` (setting) and `head_tile_coverage` (comes out of the boxes, and
# is checked separately precisely because it MUST differ here).
#
# n_fit, n_sel and n_val are in the list although they are counts: they follow
# from the splits file, so a difference means the two arms were not cut from
# the same data at all.
GLEICH = ["epochs", "seed", "cam_n", "balance_view", "balance_strength",
          "head", "head_grid", "head_negatives", "head_lambda_measured",
          "dml_index", "device_name", "n_fit", "n_sel", "n_val"]

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


def pfade_pruefen(d: pd.DataFrame, tag: str) -> None:
    """Schloss 1: die Selbstauskunft. Sagt der Lauf, dass er den Zuschnitt sah?

    Ohne die Spalten `images` und `csv` stammt die Zeile aus einer Fassung von
    rsna_train.py VOR dem 07.08.2026. Dann ist der einzige Hebel dieser Phase
    nirgends belegt, und der Vergleich darf nicht stattfinden. Das ist kein
    Formfehler: ein Arm auf den falschen Bildern saehe wie ein sauberes
    Nullergebnis aus.
    """
    for spalte in ("images", "csv"):
        if spalte not in d.columns:
            abbruch(f"results_rsna.csv hat keine Spalte '{spalte}'. Der Arm "
                    f"'{tag}' stammt aus einer Fassung von rsna_train.py vor "
                    f"dem 07.08.2026, in der der Bildordner nirgends "
                    f"protokolliert wurde. Genau das ist der Hebel dieser "
                    f"Phase; ohne ihn ist der Vergleich nicht zu belegen.")
        werte = [str(x) for x in d[spalte]]
        if any(w in ("", "nan") for w in werte):
            abbruch(f"'{tag}' hat {spalte} leer. Dieser Fold wurde vor der "
                    f"Provenienzspalte gerechnet.")
        schlecht = [w for w in werte if CROP_ORDNER not in w.replace("/", "\\")]
        if schlecht:
            abbruch(f"'{tag}' hat {spalte} = {sorted(set(schlecht))}, erwartet "
                    f"war ein Pfad mit '{CROP_ORDNER}'. DIESER ARM SAH NICHT "
                    f"DEN ZUSCHNITT. Der Vergleich unten waere ein Vergleich "
                    f"zweier identischer Rezepte und saehe wie ein sauberes "
                    f"Nullergebnis aus.")
    # Bilder und Kaesten muessen DENSELBEN Ordner nennen, sonst trainiert der
    # Kopf auf Kaesten aus einem anderen Koordinatensystem. rsna_train.py
    # bricht dagegen selbst ab, seit 07.08.; alte Zeilen tun es nicht.
    for i, r in d.iterrows():
        a = str(r["images"]).replace("/", "\\").rstrip("\\")
        b = str(r["csv"]).replace("/", "\\").rstrip("\\")
        if a != b:
            abbruch(f"'{tag}' Fold {r['fold']}: images = {a}, csv = {b}. Die "
                    f"Kaesten gehoeren dann zu einem anderen Bild als die "
                    f"Bildpunkte.")


def ein_hebel_pruefen(arm: pd.DataFrame, bez: pd.DataFrame) -> None:
    """Schloss 2: unterscheidet sich der Arm WIRKLICH nur im Zuschnitt?

    Gepaart heisst: nur eine Sache darf sich unterscheiden. Diese Pruefung
    schreibt das nicht vor, sie liest es nach. Sie faengt den Fall, in dem
    beim Zusammenbauen der Befehlszeile ausser dem Bildordner noch etwas
    anderes verrutscht ist, etwa der Adapter oder die Zahl der Epochen. Genau
    solche Zweitunterschiede sind hinterher nicht mehr auffindbar, weil sie in
    keiner Ueberschrift stehen.
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
        abbruch("die beiden Arme unterscheiden sich in mehr als im "
                "Zuschnitt:\n    " + "\n    ".join(fehler) +
                "\n  Ein gepaarter Vergleich mit zwei Unterschieden "
                "beantwortet keine Frage.")
    print(f"  ok   beide Arme stimmen in {len(GLEICH)} Konfigurationsspalten "
          f"ueberein")


def staerke_pruefen(d: pd.DataFrame, tag: str, soll: dict,
                    darf_leer: bool) -> None:
    """Die Augmentierungsstaerke, geprueft statt angenommen.

    In Phase 6 stand hier die NEUE Staerke. In Phase 7 steht hier die ALTE,
    und die Umkehrung ist der ganze Punkt: Phase 6 ist durchgefallen, ihr Arm
    ist nicht der Bezug. Ein Phase-7-Arm, der die Phase-6-Staerke
    mitgeschleppt haette, bewegte zwei Dinge gleichzeitig.
    """
    for spalte, wert in soll.items():
        if spalte not in d.columns:
            if darf_leer:
                continue
            abbruch(f"'{tag}' hat keine Spalte {spalte}.")
        ist = d[spalte].to_numpy(dtype=float)
        if np.all(np.isnan(ist)):
            if darf_leer:
                continue
            abbruch(f"'{tag}' hat {spalte} leer. Die Staerke ist nicht belegt.")
        if not np.allclose(ist, wert, atol=1e-9, equal_nan=False):
            abbruch(f"'{tag}' hat {spalte} = {sorted(set(ist))}, erwartet "
                    f"{wert}. Phase 7 laeuft mit der ALTEN "
                    f"Augmentierungsstaerke; alles andere bewegt zwei Dinge "
                    f"auf einmal.")


def abdeckung_pruefen(d: pd.DataFrame, bez: pd.DataFrame) -> None:
    """Schloss 3: der Beleg aus den Daten, nicht aus einer Selbstauskunft.

    `head_tile_coverage` ist der Anteil der Bildflaeche, den die Kaesten der
    Fit-Bilder bedecken. Er wird aus den Kaesten in --csv gerechnet, also aus
    dem, was der Lauf TATSAECHLICH geladen hat. Der feste Zuschnitt
    vergroessert jeden Kasten um 1/0,80 linear, die Abdeckung muss also um
    rund 1,554 steigen. Der Sollwert je Fold stand vor dem Lauf fest.

    Der Unterschied zu den Pfadspalten: die kann eine falsche Kommandozeile
    konsistent falsch fuellen. Diese Zahl nicht.
    """
    if "head_tile_coverage" not in d.columns:
        abbruch("results_rsna.csv hat keine Spalte head_tile_coverage.")
    print(f"\n  {'Fold':>5}{'Phase 5':>10}{'Phase 7 ist':>14}"
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
                f"  Bei {alt:.4f} laege sie, wenn der Lauf die ALTEN Kaesten "
                f"geladen haette; das ist der teuerste Fehler dieser Phase.\n"
                f"  Bei rund 0,164 laege sie bei fester Seite 0,85 statt 0,80.\n"
                f"  Die Sollwerte stehen in erklaerungen/"
                f"23_phase7_zuschnitt.md, Abschnitt 12.")
    print("  ok   die Kastenabdeckung ist gestiegen wie vorhergesagt. Der Arm")
    print("       hat die Kaesten DES ZUSCHNITTS geladen, nicht die alten.")


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


def zuschnitt_nachrechnen(pfad: Path) -> None:
    """Der Zuschnitt selbst, aus seiner Parameterdatei nachgerechnet.

    Dieselbe Regel eine Ebene tiefer: der Bericht von rsna_make_crops.py wurde
    von dem Skript gedruckt, das den Zuschnitt erzeugt hat. Hier wird er aus
    den gespeicherten Fensterparametern neu gebildet, von einem Skript, das
    ihn nicht erzeugt hat.
    """
    if not pfad.exists():
        print(f"  Hinweis: {pfad} fehlt, der Zuschnitt wird nicht "
              f"nachgerechnet.")
        return
    p = pd.read_csv(pfad)
    spanne = float(p["side"].max() - p["side"].min())
    print(f"  Fenster aus {pfad.name}: {len(p)} Bilder, Seite "
          f"{p['side'].median():.3f}, Spanne {spanne:.6f}")
    # Der Versatz steht seit 07.08. als konstante Spalte in der Parameterdatei.
    # Vorher war er nach dem Lauf ueberhaupt nicht mehr feststellbar: `top`
    # mischt Maskenmitte und Versatz, und aus der Summe laesst sich der
    # Summand nicht zurueckholen. Eine Konstante in der Vorfestlegung, die kein
    # Codepfad liest, schuetzt nichts.
    if "shift_y" not in p.columns:
        abbruch(f"{pfad.name} hat keine Spalte shift_y. Die Datei stammt aus "
                f"einer Fassung von rsna_make_crops.py vor dem 07.08.2026, und "
                f"dann laesst sich der Versatz nachtraeglich nicht mehr "
                f"belegen. Den Zuschnitt mit der aktuellen Fassung neu "
                f"erzeugen.")
    v_spanne = float(p["shift_y"].max() - p["shift_y"].min())
    v = float(p["shift_y"].iloc[0])
    if v_spanne != 0.0:
        abbruch(f"shift_y streut um {v_spanne:.6f}. Der Versatz MUSS fuer "
                f"jedes Bild derselbe sein, sonst kodiert er etwas je Bild "
                f"und ist ein neuer Confounder statt einer Korrektur.")
    if abs(v - VERSATZ_Y) > 1e-9:
        abbruch(f"der Versatz ist {v}, vorfestgelegt war {VERSATZ_Y}.")
    print(f"  Versatz nach unten {v:.3f}, fuer jedes Bild derselbe")
    if spanne != 0.0:
        abbruch(f"die Seitenlaenge streut um {spanne:.6f}. Dann leitet noch "
                f"ein Pfad die Fenstergroesse aus dem Bild ab, und dieser Lauf "
                f"misst das ADAPTIVE Fenster unter neuem Namen. Genau das war "
                f"der Fehlschlag vom 26.07.")
    if abs(float(p["side"].median()) - FESTE_SEITE) > 1e-9:
        abbruch(f"die Seitenlaenge ist {p['side'].median():.4f}, "
                f"vorfestgelegt war {FESTE_SEITE}.")
    mit = p.dropna(subset=["box_frac"])
    if len(mit):
        print(f"  Boxerhalt {mit['box_frac'].mean():.4f} im Mittel, "
              f"{int((mit['box_frac'] < 0.90).sum())} Bilder unter 90 %, "
              f"{int((mit['box_frac'] <= 0.0).sum())} Kasten ganz verloren")
        print(f"  Kaesten {int(mit['n_boxes_vorher'].sum())} vorher, "
              f"{int(mit['n_boxes_nachher'].sum())} nachher")
    print("  ok   konstante Fenstergroesse, der Zoom traegt keine Information "
          "je Bild")


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------

def urteil_c(r: dict) -> tuple[str, str]:
    """C must FALL. A rise is the problem, so the UPPER end decides."""
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


def zeige(name: str, r: dict, spruch: tuple[str, str],
          erwartet: float | None = None) -> None:
    print(f"\n  {name}")
    print("    je Fold  " + "  ".join(f"{x:+.4f}" for x in r["je_fold"]))
    print(f"    Differenz {r['mean']:+.4f} +- {r['sd']:.4f}   "
          f"{int(NIVEAU * 100)}-Prozent-Intervall "
          f"[{r['lo']:+.4f}, {r['hi']:+.4f}]")
    # Die Erwartung stand nur fuer C in der Vorfestlegung. Auf A gehoert sie
    # nicht daneben: dort ist die Streuung sechsmal kleiner, und eine Zahl aus
    # der falschen Zeile neben einem Ergebnis liest sich wie ein Urteil.
    hinweis = ("" if erwartet is None
               else f"   (vorher erwartet: rund {erwartet:.3f})")
    print(f"    Halbbreite {0.5 * (r['hi'] - r['lo']):.4f}{hinweis}")
    print(f"    {spruch[0]}: {spruch[1]}")


# --------------------------------------------------------------------------
# Descriptive part
# --------------------------------------------------------------------------

def stoerungen_zeigen(bez: pd.DataFrame, arm: pd.DataFrame) -> None:
    block("BESCHREIBEND 1: C unter Stoerung des Testbildes")
    print("  Nicht vorfestgelegt. Die Frage ist, WORIN der Projektionshinweis")
    print("  steckt. Phase 6 fand: keine geometrische Stoerung senkt C, nur")
    print("  die Helligkeit. Wenn das stimmt, aendert der Zuschnitt daran")
    print("  wenig, und dann ist diese Tabelle die Begruendung fuer den")
    print("  photometrischen Arm.")
    print()
    print(f"  {'Stoerung':<12}{'Phase 5':>10}{'Phase 7':>10}"
          f"{'P5 gg clean':>13}{'P7 gg clean':>13}")
    b0 = bez["C_clean"].mean()
    a0 = arm["C_clean"].mean()
    for s in STOERUNGEN:
        k = "C_" + s[2:]
        b, a = bez[k].mean(), arm[k].mean()
        mark = "  <-- Rahmung" if s in GEOMETRISCH else ""
        print(f"  {s[2:]:<12}{b:>10.4f}{a:>10.4f}{b - b0:>+13.4f}"
              f"{a - a0:>+13.4f}{mark}")


def kanal_zeigen(pfad: Path, restkanal: Path) -> None:
    block("BESCHREIBEND 2: wie viel vom Geometriekanal nahm der Zuschnitt?")
    if not pfad.exists():
        print(f"  {pfad} fehlt, uebersprungen.")
        return
    t = pd.read_csv(pfad)
    zeilen = {"IST, wie Phase 5 trainiert": "IST: hull d8 pad.05",
              "FIX 0,80, dieser Arm": "FIX .80",
              "FIX 0,85, nicht gewaehlt": "FIX .85 (bbox-Mitte)"}
    print("  Teil 1, die Geometrie DES FENSTERS. Gemessen vor jedem Training,")
    print("  aus qc/crop_varianten_tabelle.csv. Als Abstand zur Muenze")
    print("  gelesen, weil die Richtung nichts bedeutet.")
    print()
    print(f"  {'Variante':<28}{'Fenster':>10}{'Geometrie':>12}"
          f"{'Abstand':>10}{'Boxerhalt':>12}")
    for name, schl in zeilen.items():
        r = t[t["Variante"] == schl]
        if r.empty:
            continue
        r = r.iloc[0]
        g = float(r["AUC_Geometrie_zu_APPA"])
        print(f"  {name:<28}{float(r['AUC_Fenstergroesse_zu_APPA']):>10.4f}"
              f"{g:>12.4f}{abs(g - 0.5):>10.4f}"
              f"{float(r['Boxflaeche_erhalten']):>12.4f}")
    ist = t[t["Variante"] == "IST: hull d8 pad.05"]
    fix = t[t["Variante"] == "FIX .80"]
    if not ist.empty and not fix.empty:
        a = abs(float(ist.iloc[0]["AUC_Geometrie_zu_APPA"]) - 0.5)
        b = abs(float(fix.iloc[0]["AUC_Geometrie_zu_APPA"]) - 0.5)
        print()
        print(f"  Von der Geometrie DES FENSTERS nimmt der Zuschnitt "
              f"{100 * (1 - b / a):.0f} Prozent.")
        print("  Phase 6 nahm 24 Prozent des Groessenhinweises und bewegte C "
              "um 0,0052.")

    # Teil 2. Die Frage, die bis zum 07.08. niemand gestellt hatte: das Modell
    # sieht nie Fensterparameter, es sieht Bildpunkte. Ohne diesen Block liest
    # sich Teil 1 wie "der Kanal ist zu drei Vierteln weg", und das gilt fuer
    # das Fenster, nicht fuer das Bild.
    print()
    print("  Teil 2, die Geometrie IM BILD. Das Modell bekommt nie die")
    print("  Fensterparameter, es bekommt Bildpunkte, und darin liegt eine")
    print("  Lunge. Gerechnet von rsna_restkanal.py, geeicht an den beiden")
    print("  Zeilen oben.")
    print()
    if not restkanal.exists():
        print(f"  {restkanal} fehlt. Nachzurechnen mit:")
        print("    python rsna\\befunde\\rsna_restkanal.py")
        return
    r = pd.read_csv(restkanal).set_index("rahmen")
    try:
        g = float(r.loc["ganzes Bild", "abstand"])
        z = float(r.loc["Zuschnitt", "abstand"])
    except KeyError:
        print(f"  {restkanal} hat nicht die erwarteten Zeilen, uebersprungen.")
        return
    print(f"  {'Rahmen':<40}{'Abstand':>10}")
    print(f"  {'ganzes Bild (wie Phase 5 trainiert)':<40}{g:>10.4f}")
    print(f"  {'Zuschnitt (dieser Arm)':<40}{z:>10.4f}")
    print(f"\n  Davon nimmt der Zuschnitt {100 * (1 - z / g):.0f} Prozent, nicht "
          f"{100 * (1 - b / a):.0f}.")
    print("  Der Rest ist Anatomie und keine Rahmung: eine liegend")
    print("  aufgenommene Lunge sieht auch in einem genormten Rahmen anders")
    print("  aus. Ein Nullergebnis auf C entlastet deshalb die RAHMUNG und")
    print("  nicht die Geometrie insgesamt.")


def phase6_daneben(res: pd.DataFrame, n_bez: pd.DataFrame,
                   n_arm: pd.DataFrame) -> None:
    block("BESCHREIBEND 3: Phase 6 und Phase 7 am selben Anker")
    d = res[res["tag"] == "_p6aug"]
    if d.empty:
        print("  kein Phase-6-Arm in results_rsna.csv, uebersprungen.")
        return
    d = d.drop_duplicates(subset="fold", keep="last").sort_values("fold")
    if list(d["fold"]) != list(n_bez["fold"]):
        print("  Phase 6 deckt andere Folds ab, kein gepaarter Vergleich.")
        return
    print("  Beide Arme haengen am selben Bezugsarm, sind also untereinander")
    print("  lesbar. Beschreibend: fuer Phase 6 war das das vorfestgelegte")
    print("  Urteil, hier steht es nur zum Vergleich.")
    print()
    print(f"  {'Arm':<28}{'A':>9}{'dA':>10}{'C':>9}{'dC':>10}")
    print(f"  {'Bezug (Phase 5)':<28}{n_bez['A'].mean():>9.4f}{'':>10}"
          f"{n_bez['C'].mean():>9.4f}")
    for name, a, c in (("Phase 6, Augmentierung",
                        float(d["auc_stratified"].mean()),
                        float(d["auc_view"].mean())),
                       ("Phase 7, fester Zuschnitt",
                        float(n_arm["A"].mean()), float(n_arm["C"].mean()))):
        print(f"  {name:<28}{a:>9.4f}{a - n_bez['A'].mean():>+10.4f}"
              f"{c:>9.4f}{c - n_bez['C'].mean():>+10.4f}")


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ergebnisse", type=Path, default=Path("results_rsna.csv"))
    p.add_argument("--arm-dir", type=Path, default=Path(ARM_DIR))
    p.add_argument("--bezug-dir", type=Path, default=Path(BEZUG_DIR))
    p.add_argument("--crop-params", type=Path,
                   default=Path("predictions_rsna/crop_params_fix080.csv"))
    p.add_argument("--varianten", type=Path,
                   default=Path("qc/crop_varianten_tabelle.csv"))
    p.add_argument("--restkanal", type=Path,
                   default=Path("predictions_rsna/restkanal.csv"),
                   help="Ausgabe von rsna_restkanal.py; fehlt sie, wird "
                        "Teil 2 des beschreibenden Blocks uebersprungen")
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

    # --- die drei Schloesser um den einen Hebel ---------------------------
    pfade_pruefen(r_arm, ARM_TAG)
    print(f"  ok   '{ARM_TAG}' trainierte auf {sorted(set(r_arm['images']))[0]}")
    ein_hebel_pruefen(r_arm, r_bez)
    staerke_pruefen(r_arm, ARM_TAG, SOLL_AUG, darf_leer=False)
    print(f"  ok   '{ARM_TAG}' trainierte mit der ALTEN Augmentierung "
          f"(Verschiebung {SOLL_AUG['aug_translate']}, Skalierung "
          f"{SOLL_AUG['aug_scale_lo']} bis {SOLL_AUG['aug_scale_hi']})")
    staerke_pruefen(r_bez, BEZUG_TAG, SOLL_AUG, darf_leer=True)
    print(f"  ok   '{BEZUG_TAG}' hat dieselbe Staerke oder stammt aus der "
          f"Zeit vor den Schaltern")
    abdeckung_pruefen(r_arm, r_bez)

    block("DER ZUSCHNITT, aus seinen Fensterparametern nachgerechnet")
    zuschnitt_nachrechnen(args.crop_params)

    block("DIE ENDPUNKTE, aus den Vorhersagen je Bild nachgerechnet")
    n_arm = nachrechnen(args.arm_dir, folds)
    n_bez = nachrechnen(args.bezug_dir, folds)
    herkunft("Phase 7", r_arm, n_arm)
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

    block("DAS URTEIL, nach der Vorfestlegung vom 07.08.2026")
    print(f"  Anker aus {BEZUG_DIR} ueber {len(folds)} Folds: "
          f"A {a_anker:.4f}, C {c_anker:.4f}")
    print(f"  Phase 7:  A {n_arm['A'].mean():.4f}, C {n_arm['C'].mean():.4f}")

    rc = gepaart(n_arm["C"].to_numpy(), n_bez["C"].to_numpy())
    ra = gepaart(n_arm["A"].to_numpy(), n_bez["A"].to_numpy())
    sc = urteil_c(rc)
    sa = urteil_a(ra)
    zeige("PRIMAER   C, AUC(Score -> Projektion), muss FALLEN", rc, sc,
          erwartet=0.025)
    zeige(f"NEBENBED  A, geschichtete AUC, Marge {MARGE_A}", ra, sa)

    block("PHASE 7 INSGESAMT")
    print(f"  primaer      C   {sc[0]}")
    print(f"  Nebenbedingung A {sa[0]}")
    print()
    if sa[0] != "BESTANDEN":
        print("  DURCHGEFALLEN an der Nebenbedingung. Der Satz lautet: BEI")
        print("  ACHT EPOCHEN kostet dieser Zuschnitt Trennschaerfe, NICHT")
        print("  'Zuschnitt hilft nicht'. C wird oben berichtet, ist hier aber")
        print("  nachrangig: ein Arm, der schlechter trennt, hat auch weniger")
        print("  zu verraten. Erlaubter Folgeversuch: groessere Seitenlaenge,")
        print("  mit eigener Vorfestlegung.")
    elif sc[0] == "BESTANDEN":
        print("  BESTANDEN. Der feste Zuschnitt senkt den Projektionskanal,")
        print("  ohne Trennschaerfe zu kosten. Die Rahmung trug den Hinweis")
        print("  also mit, und der Zuschnitt ist das Mittel dagegen. Phase 9")
        print("  (Zuschnitt x Aufloesung) gewinnt an Wert, und der Zuschnitt")
        print("  ist Kandidat fuer das Endrezept in Phase 10.")
    elif sc[0] == "GRAUZONE":
        print("  GRAUZONE, und das war der vorhergesagte wahrscheinlichste")
        print("  Ausgang. Das Intervall enthaelt die Null, der Punktwert liegt")
        print(f"  bei {rc['mean']:+.4f}, also auf oder unter der vorher")
        print(f"  festgelegten Schwelle {GRAUZONE:+.3f}.")
        print()
        print("  DER VORFESTGELEGTE FOLGEVERSUCH: drei Keime je Fold in BEIDEN")
        print("  Armen, sonst nichts geaendert. 15 gepaarte Einheiten,")
        print("  Halbbreite auf C dann rund 0,013, rund 14 Stunden. Er ist")
        print("  dieselbe Frage genauer gestellt und wird als Fortsetzung")
        print("  berichtet, nicht als neuer Versuch.")
    elif rc["lo"] > 0:
        print("  DURCHGEFALLEN, und zwar in der Richtung des adaptiven")
        print("  Zuschnitts: C ist gesichert GESTIEGEN. Zweiter Fehlschlag")
        print("  derselben Art. Die Zuschnittachse ist geschlossen, Phase 9")
        print("  entfaellt.")
    else:
        print("  DURCHGEFALLEN am primaeren Endpunkt.")
        print()
        print("  Entlastet ist damit die RAHMUNG: von der Geometrie des")
        print("  Fensters sind drei Viertel weg und C bewegt sich nicht.")
        print("  NICHT entlastet ist die Geometrie insgesamt. Nach dem")
        print("  Zuschnitt stehen im Bild noch 0,1638 von 0,2610 Abstand zur")
        print("  Muenze, gemessen mit rsna_restkanal.py; der Rest ist Anatomie")
        print("  statt Rahmung und geometrisch nicht abzuraeumen, ohne den")
        print("  Befund mit abzuraeumen.")
        print()
        print("  Der naechste Arm ist trotzdem der photometrische: dorthin")
        print("  zeigen die Stoerungsproben der Phase 6.")
        print()
        print(f"  Der Punktwert {rc['mean']:+.4f} liegt ueber der")
        print(f"  Grauzonenschwelle {GRAUZONE:+.3f}, der teure Folgeversuch")
        print("  mit drei Keimen ist also NICHT ausgeloest.")

    # Nur wenn die Null WIRKLICH im Intervall liegt. Ohne die letzte Bedingung
    # erschiene dieser Absatz auch im Ast "C ist gesichert gestiegen" und
    # behauptete dort, das Intervall halte einen Rueckgang offen, waehrend es
    # jeden Rueckgang ausschliesst. Ein Skript, das seinem eigenen Urteil drei
    # Zeilen spaeter widerspricht, ist schlimmer als eines, das schweigt.
    if (sa[0] == "BESTANDEN" and sc[0] != "BESTANDEN"
            and rc["lo"] <= 0 <= rc["hi"]):
        halb = 0.5 * (rc["hi"] - rc["lo"])
        print()
        print("  'Nicht gefallen' heisst NICHT 'gesichert kein Effekt'. Das")
        print(f"  Intervall reicht von {rc['lo']:+.4f} bis {rc['hi']:+.4f}, es")
        print("  haelt einen Rueckgang genauso offen wie einen Anstieg. Diese")
        print(f"  Folds loesen auf C {halb:.4f} auf; die Vorfestlegung hatte")
        print("  rund 0,025 erwartet und genau deshalb die Grauzone.")

    if not args.nur_urteil:
        stoerungen_zeigen(n_bez, n_arm)
        kanal_zeigen(args.varianten, args.restkanal)
        phase6_daneben(res, n_bez, n_arm)
        block("WAS DER BESCHREIBENDE TEIL NICHT DARF")
        print("  Er darf das Urteil oben nicht umdeuten. Er sagt, WAS als")
        print("  Naechstes zu messen waere, nicht, wie dieser Lauf")
        print("  auszugehen hatte.")


if __name__ == "__main__":
    main()
