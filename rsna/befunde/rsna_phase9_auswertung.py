"""Phase 9: the verdict on the photometric arm, recomputed from scratch.

WHY THIS SCRIPT EXISTS AT ALL
-----------------------------
The training run already prints an AUC per fold. This script does not trust
those lines. It recomputes both endpoints from the per image prediction files,
compares the result against the numbers in `results_rsna.csv`, and refuses to
print a verdict if the two disagree. That is the standing rule of this project:
the conclusions of a phase get recomputed by a script that did not produce
them.

THE LEVER, AND THE LOCKS AROUND IT
-----------------------------------
Phase 9 turns up `--aug-brightness` and `--aug-contrast` from 0.15 to 0.60 and
changes nothing else. So the locks have to hold everything else still:

  1. `images` and `csv` IDENTICAL in both arms. A different image folder would
     make this a comparison between two datasets.
  2. `size` = 224 in BOTH arms, and `input_px` equal to it. Phase 8 closed the
     resolution axis; this arm runs where the photometric channel is
     strongest.
  3. `aug_translate`, `aug_scale`, `aug_degrees` at the OLD values in both
     arms. Phase 6 failed; its strength is not the reference. An arm carrying
     both the phase 6 geometry and the phase 9 photometry would move two things
     at once.
  4. `aug_brightness` and `aug_contrast` at 0.60 in the arm and 0.15 in the
     reference, where empty means 0.15.
  5. `aug_brightness_measured` and `aug_contrast_measured` have to agree with
     the switch. That pair is the `size`/`input_px` pattern of phase 8: the
     first two columns are a self report, the last two are the spread of the
     factors the transform in the loader really drew. Until 09.08.2026 the
     photometric strength sat hard wired inside `TrainTransform` while the
     three geometric strengths were already arguments, which is exactly the
     shape `--balance-strength` had when it was read and not passed on.
  6. `head_tile_coverage` UNCHANGED against phase 5. Catches a run on the crop,
     where it sits near 0.18.

THE PRE-REGISTRATION, COPIED FROM erklaerungen/27_phase9_photometrisch.md
-------------------------------------------------------------------------
Written before the run. Anchors are the five fold means of the phase 5 winner
`predictions_final_model`, the same anchors phases 6, 7 and 8 used:

    A = 0.8368    stratified AUC          finds pneumonia, should RISE
    C = 0.7467    AUC(score -> view)      gives away the acquisition, should FALL

ONE PRIMARY ENDPOINT, C. The upper end of the paired 90 percent interval has to
    lie below zero. No minimum difference, exactly as in phases 6 and 7, so
    that the four arms can be read against each other.

THE BOLT. A passed gate only counts if A is NON INFERIOR, lower end above
    -0.01. Otherwise the sentence is "a worse discriminator has less to give
    away", not "the jitter clears the confounder".

GREY ZONE. Zero inside the interval and a point estimate at or below -0.015
    triggers the pre-registered follow up with three seeds per fold in both
    arms, 15 paired units, about 14 hours at 224 px.

WHY THE KNOB IS 0.60 AND NOT 0.15
----------------------------------
Measured before the run by `rsna_photometrie_reichweite.py` on all 22872
development images, and the target was written into that script before the
numbers existed: a knob that removes less than half of the cue would be phase 6
a second time.

    global brightness   AUC -> view 0.4604, distance from the coin 0.040
    global contrast     AUC -> view 0.2420, distance from the coin 0.258

The strong global channel is the CONTRAST, not the brightness. At the default
0.15 the jitter removes 4 percent of the first and 22 percent of the second, so
every run of this project so far had, for this purpose, close to no jitter at
all. At 0.60 it removes 64 and 74 percent.

AND THE LIMIT, WRITTEN DOWN BEFORE AND NOT AFTER: that measurement is about the
GLOBAL grey value. Phase 8 showed the channel can move when it is drained in
one place. Reach is necessary, not sufficient.

USAGE
-----
    python rsna\\befunde\\rsna_phase9_auswertung.py
    python rsna\\befunde\\rsna_phase9_auswertung.py --nur-urteil
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
MARGE_A = 0.01           # non inferiority margin, the bolt on the gate
NIVEAU = 0.90            # two sided, so the one sided bound is at 5 percent
GRAUZONE = -0.015        # point estimate at or below this: the follow up runs

ARM_TAG = "_p9photo"
ARM_DIR = "predictions_p9_photo"
BEZUG_TAG = "_p5head_ex"
BEZUG_DIR = "predictions_final_model"

# Der Hebel. Beide Arme laufen bei 224 px, die Aufloesungsachse ist mit
# Phase 8 geschlossen.
ARM_PHOTO = 0.60
BEZUG_PHOTO = 0.15
SOLL_SIZE = 224

# Die gemessene Staerke gegen den Schalter. Weit, und der Grund steht im
# Docstring von `gemessene_jitter_staerke`: der Kontrast kommt am gerundeten
# Testbild systematisch 6 bis 9 Prozent zu hoch heraus. Es ist eine
# Verkabelungspruefung und keine Kalibrierung; sie muss 0,15 von 0,60
# unterscheiden, und das ist ein Faktor vier.
PHOTO_TOLERANZ_REL = 0.25
PHOTO_TOLERANZ_ABS = 0.03

SOLL_IMAGES = "png512"
SOLL_CSV = "rsna"

# Die ALTE geometrische Staerke, in beiden Armen. Phase 6 ist durchgefallen,
# ihr Arm ist nicht der Bezug.
SOLL_AUG = dict(aug_translate=0.03, aug_scale_lo=0.93, aug_scale_hi=1.07,
                aug_degrees=7.0)

# Wie in Phase 8, und aus demselben Grund: faengt einen Lauf auf dem Zuschnitt
# (dort rund 0,18). Ueber die photometrische Staerke sagt diese Zahl nichts.
SOLL_COVERAGE = {0: 0.118765, 1: 0.117498, 2: 0.117710,
                 3: 0.116605, 4: 0.117179}
COVERAGE_TOLERANZ = 0.002

ANKER_TOLERANZ = 5e-5
HERKUNFT_TOLERANZ = 1e-9

# Konfigurationsspalten, die in beiden Armen GLEICH sein muessen. Die vier
# photometrischen Spalten stehen bewusst NICHT hier: sie sind der Hebel.
GLEICH = ["epochs", "seed", "cam_n", "balance_view", "balance_strength",
          "head", "head_grid", "head_negatives", "head_lambda_measured",
          "dml_index", "device_name", "n_fit", "n_sel", "n_val"]

STOERUNGEN = ["p_clean", "p_corners", "p_zoom_in", "p_shift", "p_rotate",
              "p_low_contr", "p_bright", "p_blur", "p_lowres"]
FESTER_FAKTOR = {"p_low_contr", "p_bright"}

# Die Vorflugmessung vom 09.08., damit sie im Bericht neben dem Ergebnis steht
# und nicht nur in einer Erinnerung.
REICHWEITE = dict(auc_mittel=0.4604, auc_streuung=0.2420,
                  uebrig_mittel_015=0.96, uebrig_streuung_015=0.78,
                  uebrig_mittel_060=0.36, uebrig_streuung_060=0.26)


# --------------------------------------------------------------------------
# Statistics, written here rather than imported, so that a second
# implementation exists
# --------------------------------------------------------------------------

def rank_auc(score, label) -> float:
    """AUC as the rank statistic. Ties get their mean rank."""
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    if label.all() or not label.any():
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    n1 = int(label.sum())
    n0 = int((~label).sum())
    return float((r[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def stratified_auc(score, label, view) -> float:
    """The n weighted mean of the AP only and the PA only AUC."""
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
    """Bisection on t_cdf. scipy is not a dependency of this repo."""
    lo, hi = -60.0, 60.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gepaart(neu: np.ndarray, alt: np.ndarray, niveau: float = NIVEAU) -> dict:
    """Paired per fold: neu minus alt, one difference per fold."""
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


def _menge(v: np.ndarray) -> list:
    """Die vorkommenden Werte, lesbar. `sorted(set(...))` auf einem
    numpy-Array druckt `np.float64(512.0)`, und das in einer
    Abbruchmeldung zu lesen kostet Sekunden, die niemand hat."""
    return sorted({round(float(x), 6) for x in v if np.isfinite(x)})


def _spalte(d: pd.DataFrame, name: str) -> np.ndarray:
    if name not in d.columns:
        return np.full(len(d), np.nan)
    return pd.to_numeric(d[name], errors="coerce").to_numpy(dtype=float)


def pfade_pruefen(d: pd.DataFrame, tag: str, darf_leer: bool = False) -> bool:
    """Schloss 1: beide Arme sehen DIESELBEN Dateien.

    `darf_leer` gilt nur fuer den Bezugsarm `_p5head_ex`, der vor den
    Provenienzspalten lief. Der staerkere Beleg fuer ihn steht ohnehin weiter
    unten: rechneten sich die Anker aus einem anderen Bildordner, waeren sie
    nicht 0,8368 und 0,7467.
    """
    for spalte, soll in (("images", SOLL_IMAGES), ("csv", SOLL_CSV)):
        if spalte not in d.columns:
            if darf_leer:
                print(f"  Hinweis: '{tag}' hat keine Spalte '{spalte}'. Der "
                      f"Bezugsarm ist aelter als der 07.08.2026; belegt wird "
                      f"das ueber die Anker.")
                return False
            abbruch(f"results_rsna.csv hat keine Spalte '{spalte}'.")
        werte = [str(x) for x in d[spalte]]
        if any(w in ("", "nan") for w in werte):
            if darf_leer:
                print(f"  Hinweis: '{tag}' hat {spalte} leer. Der Bezugsarm "
                      f"lief vor der Provenienzspalte; leer heisst die "
                      f"Vorgabe, und das ist, was Phase 9 verlangt. Belegt "
                      f"wird das ueber die Anker weiter unten.")
                return False
            abbruch(f"'{tag}' hat {spalte} leer.")
        schlecht = [w for w in werte
                    if soll not in w.replace("/", "\\").split("\\")[-1]]
        if schlecht:
            abbruch(f"'{tag}' hat {spalte} = {sorted(set(schlecht))}, erwartet "
                    f"war ein Pfad, der auf '{soll}' endet.")
    return True


def aufloesung_pruefen(d: pd.DataFrame, tag: str, darf_leer: bool) -> None:
    """Schloss 2: BEIDE Arme laufen bei 224 px.

    In Phase 8 war die Kantenlaenge der Hebel, hier ist sie ein Abbruchgrund.
    Die Aufloesungsachse ist geschlossen, und der photometrische Kanal ist bei
    224 px am staerksten.
    """
    s = _spalte(d, "size")
    i = _spalte(d, "input_px")
    if np.all(np.isnan(s)):
        if darf_leer:
            print(f"  Hinweis: '{tag}' hat size leer. Leer heisst 224, siehe "
                  f"rsna_train.py; belegt ist das ueber die Anker.")
            return
        abbruch(f"'{tag}' hat size leer.")
    if not np.allclose(s, SOLL_SIZE, equal_nan=False):
        abbruch(f"'{tag}' hat size = {_menge(s)}, erwartet {SOLL_SIZE}. "
                f"Phase 9 laeuft bei 224 px in BEIDEN Armen.")
    if np.all(np.isnan(i)):
        abbruch(f"'{tag}' hat input_px leer, obwohl size gesetzt ist.")
    if not np.allclose(i, s, equal_nan=False):
        abbruch(f"'{tag}': size {_menge(s)} gegen gemessenes input_px "
                f"{_menge(i)}.")
    print(f"  ok   '{tag}' lief bei {SOLL_SIZE} px, Schalter und gemessene "
          f"Kantenlaenge stimmen ueberein")


def photometrie_pruefen(d: pd.DataFrame, tag: str, soll: float,
                        darf_leer: bool) -> None:
    """Schloss 3 und 4: der Hebel selbst, einmal als Schalter und einmal
    gemessen.

    `aug_brightness` und `aug_contrast` sind die Selbstauskunft,
    `aug_brightness_measured` und `aug_contrast_measured` sind die Streuung der
    Faktoren, die die Transformation im Loader wirklich gezogen hat. Beide
    Spaltenpaare gibt es erst seit dem 09.08.2026, und genau darum gibt es sie:
    bis dahin stand die photometrische Staerke fest verdrahtet in
    `TrainTransform`, waehrend die drei geometrischen schon Argumente waren.
    """
    for schalter, gemessen in (("aug_brightness", "aug_brightness_measured"),
                               ("aug_contrast", "aug_contrast_measured")):
        s = _spalte(d, schalter)
        g = _spalte(d, gemessen)
        if np.all(np.isnan(s)):
            if darf_leer:
                print(f"  Hinweis: '{tag}' hat {schalter} leer. Leer heisst "
                      f"{BEZUG_PHOTO}, siehe rsna_train.py; der Bezugsarm lief "
                      f"vor dem Schalter.")
                continue
            abbruch(f"results_rsna.csv hat keine Spalte '{schalter}'. Der Arm "
                    f"'{tag}' stammt aus einer Fassung von rsna_train.py vor "
                    f"dem 09.08.2026, und dann ist der einzige Hebel dieser "
                    f"Phase nirgends belegt. Ein Arm, der versehentlich bei "
                    f"{BEZUG_PHOTO} gelaufen waere, saehe wie ein sauberes "
                    f"Nullergebnis aus.")
        elif not np.allclose(s, soll, atol=1e-9, equal_nan=False):
            abbruch(f"'{tag}' hat {schalter} = {_menge(s)}, erwartet "
                    f"{soll}. DIESER ARM LIEF NICHT MIT DER VORFESTGELEGTEN "
                    f"STAERKE.")
        if np.all(np.isnan(g)):
            if darf_leer:
                continue
            abbruch(f"'{tag}' hat {gemessen} leer, obwohl {schalter} gesetzt "
                    f"ist. Dann belegt nur die Selbstauskunft den Hebel, und "
                    f"das ist genau der Fall, gegen den die Spalte gebaut "
                    f"wurde.")
        schranke = max(PHOTO_TOLERANZ_REL * soll, PHOTO_TOLERANZ_ABS)
        if np.any(np.abs(g - soll) > schranke):
            abbruch(f"'{tag}': {schalter} sagt {soll}, gezogen wurde "
                    f"{_menge(g)} (Schranke {schranke:.3f}). "
                    f"Der Schalter ist nicht dort angekommen, wo er wirkt.")
    print(f"  ok   '{tag}' lief mit photometrischer Staerke {soll}, Schalter "
          f"und gezogene Faktoren stimmen ueberein")


def ein_hebel_pruefen(arm: pd.DataFrame, bez: pd.DataFrame,
                      pfade_vergleichbar: bool = True) -> None:
    """Schloss 5: unterscheidet sich der Arm WIRKLICH nur in der Photometrie?"""
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
                "Photometrie:\n    " + "\n    ".join(fehler) +
                "\n  Ein gepaarter Vergleich mit zwei Unterschieden "
                "beantwortet keine Frage.")
    print(f"  ok   beide Arme stimmen in {len(GLEICH)} Konfigurationsspalten "
          f"ueberein")
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
                        f"Phase 9 aendert die Augmentierung, nicht die "
                        f"Dateien.")
    print("  ok   beide Arme zeigen auf dieselben Bild- und Kastenordner")


def geometrie_pruefen(d: pd.DataFrame, tag: str, darf_leer: bool) -> None:
    """Schloss 6: die GEOMETRISCHE Staerke ist die alte, in beiden Armen.

    Phase 6 ist durchgefallen, ihr Arm ist nicht der Bezug. Ein Phase-9-Arm,
    der die Phase-6-Geometrie mitgeschleppt haette, bewegte zwei Dinge auf
    einmal, und hinterher waere nicht mehr auffindbar, welches gewirkt hat.
    """
    for spalte, wert in SOLL_AUG.items():
        ist = _spalte(d, spalte)
        if np.all(np.isnan(ist)):
            if darf_leer:
                continue
            abbruch(f"'{tag}' hat {spalte} leer. Die Staerke ist nicht belegt.")
        if not np.allclose(ist, wert, atol=1e-9, equal_nan=False):
            abbruch(f"'{tag}' hat {spalte} = {_menge(ist)}, erwartet "
                    f"{wert}. Phase 9 laeuft mit der ALTEN geometrischen "
                    f"Augmentierung.")


def abdeckung_pruefen(d: pd.DataFrame, bez: pd.DataFrame) -> None:
    """Schloss 7: der Arm lief auf dem GANZEN Bild, nicht auf dem Zuschnitt.

    WAS DIESE ZAHL NICHT KANN: sie belegt die photometrische Staerke NICHT.
    Sie faengt den anderen teuren Fehler; auf crop512_fix080 laege sie bei rund
    0,18.
    """
    if "head_tile_coverage" not in d.columns:
        abbruch("results_rsna.csv hat keine Spalte head_tile_coverage.")
    print(f"\n  {'Fold':>5}{'Phase 5':>10}{'Phase 9 ist':>14}"
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
                f"gelaufen waere.")
    print("  ok   die Kastenabdeckung ist unveraendert. Der Arm lief auf dem")
    print("       GANZEN Bild. (Ueber die Photometrie sagt diese Zahl nichts,")
    print("       das belegen die vier aug-Spalten.)")


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
            zeile["A_" + s[2:]] = (stratified_auc(x[s], y, x["viewpos"])
                                   if s in x.columns else np.nan)
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
                    f"reproduzieren.")


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------

def urteil_c(r: dict) -> tuple[str, str]:
    """The gate: C must FALL. A rise is the problem, so the UPPER end decides."""
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


def urteil_a_nichtunterlegen(r: dict) -> tuple[str, str]:
    """The bolt. The LOWER end decides."""
    if r["lo"] > -MARGE_A:
        return "BESTANDEN", (f"das untere Ende liegt ueber {-MARGE_A:+.2f}, "
                             f"A ist nicht unterlegen")
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
    hinweis = ("" if erwartet is None
               else f"   (vorher erwartet: rund {erwartet:.3f})")
    print(f"    Halbbreite {0.5 * (r['hi'] - r['lo']):.4f}{hinweis}")
    print(f"    {spruch[0]}: {spruch[1]}")


# --------------------------------------------------------------------------
# Descriptive part
# --------------------------------------------------------------------------

def stoerungen_zeigen(bez: pd.DataFrame, arm: pd.DataFrame) -> None:
    block("BESCHREIBEND 1: C unter Stoerung des Testbildes")
    print("  Nicht vorfestgelegt, und mit einer Leseregel, die seit dem 09.08.")
    print("  gilt und die Tabelle der Phasen 6, 7 und 8 nachtraeglich")
    print("  entwertet:")
    print()
    print("  EINE STOERUNG MIT FESTEM FAKTOR KANN EINEN RANGKANAL NICHT")
    print("  MESSEN. low_contr multipliziert JEDES Bild mit 0,6 und bright")
    print("  jedes mit 1,35. Ein fester Faktor laesst die Reihenfolge der")
    print("  Bilder unangetastet, und eine AUC liest nur die Reihenfolge.")
    print("  Was diese beiden Zeilen ueberhaupt bewegen kann, ist allein das")
    print("  Beschneiden bei 0 und 255 und die Nichtlinearitaet des Modells.")
    print("  Sie messen Empfindlichkeit, nicht Kanalstaerke.")
    print()
    print("  WAS HIER TROTZDEM ZU ERWARTEN WAR, vor dem Lauf aufgeschrieben:")
    print("  ein Modell, das mit Faktoren von 0,4 bis 1,6 trainiert wurde,")
    print("  sollte gegen einen festen Faktor 1,35 fast unempfindlich sein.")
    print("  Der Ausschlag von bright muss also SCHRUMPFEN. Das ist ein Beleg")
    print("  dafuer, dass der Knopf gewirkt hat, und KEINER dafuer, dass C")
    print("  gefallen ist.")
    print()
    print(f"  {'Stoerung':<12}{'Phase 5':>10}{'Phase 9':>10}"
          f"{'P5 gg clean':>13}{'P9 gg clean':>13}")
    b0 = bez["C_clean"].mean()
    a0 = arm["C_clean"].mean()
    for s in STOERUNGEN:
        k = "C_" + s[2:]
        b, a = bez[k].mean(), arm[k].mean()
        mark = "  <-- fester Faktor" if s in FESTER_FAKTOR else ""
        print(f"  {s[2:]:<12}{b:>10.4f}{a:>10.4f}{b - b0:>+13.4f}"
              f"{a - a0:>+13.4f}{mark}")


def reichweite_zeigen() -> None:
    block("BESCHREIBEND 2: die Vorflugmessung, vor dem Lauf gerechnet")
    p = REICHWEITE
    print("  Aus rsna_photometrie_reichweite.py, alle 22872 Entwicklungsbilder.")
    print("  Die Zielmarke stand im Skriptkopf, bevor die Zahlen existierten:")
    print("  ein Knopf, der weniger als die Haelfte wegnimmt, waere Phase 6")
    print("  noch einmal.")
    print()
    print(f"  {'globale Groesse':<22}{'AUC -> Projektion':>19}"
          f"{'uebrig bei 0,15':>17}{'uebrig bei 0,60':>17}")
    print(f"  {'Mittelwert':<22}{p['auc_mittel']:>19.4f}"
          f"{p['uebrig_mittel_015']:>17.0%}{p['uebrig_mittel_060']:>17.0%}")
    print(f"  {'Streuung':<22}{p['auc_streuung']:>19.4f}"
          f"{p['uebrig_streuung_015']:>17.0%}{p['uebrig_streuung_060']:>17.0%}")
    print()
    print("  Der starke globale Kanal ist der KONTRAST, nicht die Helligkeit.")
    print("  Und bei 0,15, dem Wert jedes Laufs dieses Projekts bis hierher,")
    print("  ist der Jitter fuer diesen Zweck praktisch nicht vorhanden.")


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
                      ("_p7fix080", "Phase 7, fester Zuschnitt"),
                      ("_p8s512", "Phase 8, 512 px")):
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
    print(f"  {'Phase 9, Photometrie 0,60':<28}"
          f"{a:>9.4f}{a - n_bez['A'].mean():>+10.4f}"
          f"{c:>9.4f}{c - n_bez['C'].mean():>+10.4f}")


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ergebnisse", type=Path, default=Path("results_rsna.csv"))
    p.add_argument("--arm-dir", type=Path, default=Path(ARM_DIR))
    p.add_argument("--bezug-dir", type=Path, default=Path(BEZUG_DIR))
    p.add_argument("--arm-tag", default=ARM_TAG,
                   help="nur fuer Gegentests: einen ANDEREN Arm durch dieselben "
                        "Schloesser schicken und sehen, ob sie zuschnappen")
    p.add_argument("--nur-urteil", action="store_true",
                   help="ohne den beschreibenden Teil")
    args = p.parse_args()

    if not args.ergebnisse.exists():
        abbruch(f"{args.ergebnisse} fehlt.")
    res = pd.read_csv(args.ergebnisse)

    block("HERKUNFT, vor jeder Zahl")
    r_arm = zeilen_holen(res, args.arm_tag)
    r_bez = zeilen_holen(res, BEZUG_TAG)
    if list(r_arm["fold"]) != list(r_bez["fold"]):
        abbruch(f"die Arme decken verschiedene Folds ab: "
                f"{list(r_arm['fold'])} gegen {list(r_bez['fold'])}.")
    folds = list(r_arm["fold"])
    print(f"  ok   beide Arme decken dieselben Folds: {folds}")

    pfade_pruefen(r_arm, args.arm_tag)
    print(f"  ok   '{args.arm_tag}' trainierte auf "
          f"{sorted(set(r_arm['images']))[0]}")
    bez_pfade_da = pfade_pruefen(r_bez, BEZUG_TAG, darf_leer=True)
    aufloesung_pruefen(r_arm, args.arm_tag, darf_leer=False)
    aufloesung_pruefen(r_bez, BEZUG_TAG, darf_leer=True)
    photometrie_pruefen(r_arm, args.arm_tag, ARM_PHOTO, darf_leer=False)
    photometrie_pruefen(r_bez, BEZUG_TAG, BEZUG_PHOTO, darf_leer=True)
    ein_hebel_pruefen(r_arm, r_bez, pfade_vergleichbar=bez_pfade_da)
    geometrie_pruefen(r_arm, args.arm_tag, darf_leer=False)
    geometrie_pruefen(r_bez, BEZUG_TAG, darf_leer=True)
    print(f"  ok   beide Arme haben die ALTE geometrische Augmentierung "
          f"(Verschiebung {SOLL_AUG['aug_translate']}, Skalierung "
          f"{SOLL_AUG['aug_scale_lo']} bis {SOLL_AUG['aug_scale_hi']})")
    abdeckung_pruefen(r_arm, r_bez)

    block("DIE ENDPUNKTE, aus den Vorhersagen je Bild nachgerechnet")
    n_arm = nachrechnen(args.arm_dir, folds)
    n_bez = nachrechnen(args.bezug_dir, folds)
    herkunft("Phase 9", r_arm, n_arm)
    herkunft("Phase 5", r_bez, n_bez)

    a_anker = float(n_bez["A"].mean())
    c_anker = float(n_bez["C"].mean())
    for name, ist, soll in (("A", a_anker, ANKER_A), ("C", c_anker, ANKER_C)):
        if abs(ist - soll) > ANKER_TOLERANZ:
            abbruch(f"der Anker {name} rechnet sich zu {ist:.4f}, "
                    f"vorfestgelegt war {soll}. Der Bezugsarm ist nicht mehr "
                    f"der, gegen den vorfestgelegt wurde.")
        print(f"  ok   Anker {name} = {ist:.4f} wie vorfestgelegt ({soll})")

    block("DAS URTEIL, nach der Vorfestlegung vom 09.08.2026")
    print(f"  Anker aus {BEZUG_DIR} ueber {len(folds)} Folds: "
          f"A {a_anker:.4f}, C {c_anker:.4f}")
    print(f"  Phase 9:  A {n_arm['A'].mean():.4f}, C {n_arm['C'].mean():.4f}")
    print()
    print("  A misst, ob der Score die PNEUMONIE vorhersagt, und darf nicht")
    print("  unterlegen sein. C misst, ob derselbe Score die AUFNAHMEART")
    print("  vorhersagt, und ist der primaere Endpunkt: er soll FALLEN.")

    ra = gepaart(n_arm["A"].to_numpy(), n_bez["A"].to_numpy())
    rc = gepaart(n_arm["C"].to_numpy(), n_bez["C"].to_numpy())
    sc = urteil_c(rc)
    sn = urteil_a_nichtunterlegen(ra)
    zeige("PRIMAER   AUC(Score -> Projektion), muss FALLEN", rc, sc,
          erwartet=0.023)
    zeige(f"RIEGEL    geschichtete AUC, nicht unterlegen, Marge {MARGE_A}",
          ra, sn)

    block("PHASE 9 INSGESAMT")
    zaehlt = sc[0] == "BESTANDEN" and sn[0] == "BESTANDEN"
    print(f"  Primaer  {sc[0]}" +
          ("" if sn[0] == "BESTANDEN" else "   (Riegel offen, zaehlt nicht)"))
    print(f"  Riegel   {sn[0]}")
    print()
    if zaehlt:
        print("  BESTANDEN. Der photometrische Jitter senkt den")
        print("  Projektionskanal, ohne Trennschaerfe zu kosten. Das ist der")
        print("  erste Eingriff am BILD, der das schafft; die volle")
        print("  Entkopplung wirkte im Trainingsstrom, nicht am Pixel.")
        print("  Damit kommt die Staerke 0,60 als Kandidat ins Endrezept, und")
        print("  Phase 10 muss klaeren, ob sie mit der staerkeren")
        print("  Augmentierung aus Phase 6 zusammen noch etwas bringt.")
    elif sn[0] != "BESTANDEN":
        print("  DURCHGEFALLEN, und der Riegel hat gegriffen. A ist um mehr")
        print("  als die Marge gefallen. Der Satz lautet: die Staerke 0,60")
        print("  kostet Trennschaerfe. Ein C-Rueckgang waere hier nicht zu")
        print("  verwerten, denn ein schlechterer Trenner hat auch weniger zu")
        print("  verraten. Erlaubter Folgeversuch: dieselbe Achse schwaecher,")
        print("  mit eigener Vorfestlegung; die Reichweitentabelle nennt 0,40")
        print("  und 0,50 und sagt, was sie kosten.")
    elif sc[0] == "GRAUZONE":
        print("  GRAUZONE. Das Intervall enthaelt die Null, der Punktwert")
        print(f"  liegt bei {rc['mean']:+.4f}, also auf oder unter der vorher")
        print(f"  festgelegten Schwelle {GRAUZONE:+.3f}.")
        print()
        print("  DER VORFESTGELEGTE FOLGEVERSUCH: drei Keime je Fold in BEIDEN")
        print("  Armen, 15 gepaarte Einheiten, Halbbreite auf C dann rund")
        print("  0,013. Bei 224 px kostet er rund 14 Stunden. Das ist eine")
        print("  Entscheidung ueber Rechenzeit und keine ueber Statistik: die")
        print("  Schwelle stand vorher, der Preis auch.")
    elif rc["lo"] > 0:
        print("  DURCHGEFALLEN, und C ist gesichert GESTIEGEN. Das waere der")
        print("  vierte Beleg fuer [[normieren-kodiert-um]]: eine")
        print("  photometrische Aenderung, die den Kanal umkodiert statt ihn")
        print("  zu entfernen, diesmal nicht deterministisch, sondern durch")
        print("  Rauschen. Das waere ein staerkeres Ergebnis als ein blosses")
        print("  Nichts und gehoert in den Ergebnistext.")
    else:
        print("  DURCHGEFALLEN. Der Knopf hatte nachweislich Reichweite")
        print("  (64 und 74 Prozent des globalen Kanals weg, vor dem Lauf")
        print("  gemessen), und C faellt trotzdem nicht.")
        print()
        print("  Das ist der vierte Arm in Folge, der C nicht bewegt, und der")
        print("  erste, bei dem vorher belegt war, dass der Hebel greift.")
        print("  Damit ist die Lesart 'der Confounder sitzt in den Grauwerten'")
        print("  als HANDLUNGSANWEISUNG erledigt: das Modell findet die")
        print("  Projektion auch dann, wenn der globale Grauwertkanal zu drei")
        print("  Vierteln verrauscht ist. Uebrig bleibt der Trainingsstrom,")
        print("  wo die volle Entkopplung 0,0554 bewegt hat.")

    if sc[0] != "BESTANDEN" and rc["lo"] <= 0 <= rc["hi"]:
        halb = 0.5 * (rc["hi"] - rc["lo"])
        print()
        print("  'Nicht gefallen' heisst NICHT 'gesichert kein Effekt'. Das")
        print(f"  Intervall auf C reicht von {rc['lo']:+.4f} bis {rc['hi']:+.4f};")
        print(f"  diese Folds loesen {halb:.4f} auf, die Vorfestlegung hatte")
        print("  rund 0,023 erwartet.")

    if not args.nur_urteil:
        stoerungen_zeigen(n_bez, n_arm)
        reichweite_zeigen()
        vergleich_zeigen(res, n_bez, n_arm)
        block("WAS DER BESCHREIBENDE TEIL NICHT DARF")
        print("  Er darf das Urteil oben nicht umdeuten. Er sagt, WAS als")
        print("  Naechstes zu messen waere, nicht, wie dieser Lauf")
        print("  auszugehen hatte.")


if __name__ == "__main__":
    main()
