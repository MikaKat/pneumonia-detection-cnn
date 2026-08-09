"""Waechter fuer das Urteil, die Kalibrierung und die Sperre der Phase 10.

DIE STILLEN FEHLER, GEGEN DIE DIESE DATEI EXISTIERT
---------------------------------------------------
Drei Sorten, und die dritte gibt es in diesem Projekt zum ersten Mal.

Die erste ist die bekannte: das Auswertungsskript faellt ein Urteil, das nach
dem Lauf nicht mehr verhandelt wird, und ein Vorzeichen- oder Vergleichsfehler
in der Urteilsfunktion faellt niemandem auf, weil es keine zweite Meinung gibt.
Dagegen stehen unten Faelle, deren Antwort feststeht, samt dem Randfall GENAU
AUF der Latte.

Die zweite ist die Kalibrierung. Eine Platt-Kurve ist streng monoton und kann
die Reihenfolge innerhalb eines Modells nicht aendern; bewegt sich die AUC
trotzdem, ist die Verkabelung falsch, nicht die Kalibrierung. Weil zwei
Dateien dieselbe Kurve anwenden (`rsna_platt.py` beim Fitten,
`rsna_holdout.py` beim Rechnen), wird ausserdem geprueft, dass beide Umsetzungen
bitgleich dasselbe tun. Zwei Umsetzungen, die auseinanderlaufen, waeren die
leiseste Art, die App anders rechnen zu lassen als die Auswertung.

Die dritte ist neu und der Grund, warum diese Phase ueberhaupt eine eigene
Absicherung braucht: der Holdout ist eine Ressource, die sich verbraucht.
Geprueft wird deshalb, dass die Sperrdatei den Namen traegt, den die Auswertung
sucht, und dass die Auswertung ohne sie gar nicht erst rechnet.

Keine GPU, keine Bilder, kein Training.

  python tests\\test_rsna_phase10.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import _repo_path  # noqa: F401

from rsna_phase10_auswertung import (ANKER_A, ANKER_A_SD, ANKER_C, ARM_TAG,
                                     LATTE_A, NIVEAU, SOLL_N, VORHER_HI,
                                     VORHER_LO, ZIEHUNGEN, brier, ece,
                                     rank_auc, stratified_auc, urteil_a)
from rsna_platt import platt_apply, platt_fit, youden
from rsna_platt import platt_apply as platt_apply_fit
from rsna_platt import rank_auc as rank_auc_platt

FAILED = []


def check(name: str, cond: bool, info: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FAILED.append(name)


def test_tor_a():
    print(f"\nDAS TOR: unteres Ende ueber der Latte {LATTE_A}")
    faelle = [
        ("klar darueber", 0.8500, 0.8380, "BESTANDEN"),
        ("knapp darueber", 0.8100, 0.8001, "BESTANDEN"),
        ("GENAU auf der Latte zaehlt NICHT", 0.8100, 0.8000, "DURCHGEFALLEN"),
        ("knapp darunter", 0.8090, 0.7999, "DURCHGEFALLEN"),
        ("klar darunter", 0.7700, 0.7550, "DURCHGEFALLEN"),
    ]
    for name, punkt, lo, soll in faelle:
        ist, _ = urteil_a(punkt, lo)
        check(name, ist == soll, f"{ist} (erwartet {soll})")

    # Nur das UNTERE Ende entscheidet. Ein hoher Punktwert bei breitem
    # Intervall darf das Tor nicht oeffnen, sonst waere die Latte eine
    # Dekoration.
    ist, _ = urteil_a(0.8900, 0.7900)
    check("ein hoher Punktwert rettet ein zu breites Intervall nicht",
          ist == "DURCHGEFALLEN", ist)


def test_konstanten():
    print("\nDIE KONSTANTEN gegen erklaerungen/29_phase10_final.md")
    doc = Path("erklaerungen") / "29_phase10_final.md"
    if not doc.is_file():
        check("die Vorfestlegung liegt im Repo", False, str(doc))
        return
    t = doc.read_text(encoding="utf-8")

    def steht(x: str) -> bool:
        return x in t.replace("−", "-")

    for name, wert, wie in [
        ("LATTE_A", LATTE_A, "0,80"),
        ("ANKER_A", ANKER_A, "0,8368"),
        ("ANKER_A_SD", ANKER_A_SD, "0,0105"),
        ("ANKER_C", ANKER_C, "0,7467"),
        ("VORHER_LO", VORHER_LO, "0,8122"),
        ("VORHER_HI", VORHER_HI, "0,8614"),
        ("SOLL_N", SOLL_N, "3812"),
        ("ZIEHUNGEN", ZIEHUNGEN, "2000"),
    ]:
        check(f"{name} = {wert}", steht(wie), f"im Text gesucht: {wie}")
    check("NIVEAU = 0.90", NIVEAU == 0.90 and steht("90-Prozent"))
    check(f"ARM_TAG = {ARM_TAG}", ARM_TAG == "_p5head_ex" and steht("p5head_ex"))

    # Die Latte ist nicht frei gewaehlt, sondern drei Foldstreuungen unter dem
    # Anker, abgerundet. Wer eine der drei Zahlen aendert, soll hier
    # stolpern.
    drei = ANKER_A - 3 * ANKER_A_SD
    check("die Latte liegt unter 'Anker minus drei Foldstreuungen'",
          LATTE_A < drei, f"{LATTE_A} gegen {drei:.4f}")
    check("und nicht mehr als 0,01 darunter", drei - LATTE_A <= 0.01,
          f"Abstand {drei - LATTE_A:.4f}")


def test_platt():
    print("\nDIE KALIBRIERUNG")
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.random(n) < 0.25).astype(float)
    # Rohwerte mit bekanntem Versatz: das Modell ist zu selbstsicher.
    z = 1.7 * y + rng.normal(0, 1.0, n) + 0.8
    p = 1 / (1 + np.exp(-1.9 * z))

    a, b = platt_fit(p, y)
    pc = platt_apply(p, a, b)

    check("die Kurve ist monoton, die Steigung ist positiv", a > 0, f"a {a:.4f}")
    check("die Reihenfolge bleibt erhalten",
          bool(np.array_equal(np.argsort(p), np.argsort(pc))))
    a_roh = rank_auc(p, y)
    a_kal = rank_auc(pc, y)
    check("die AUC bewegt sich nicht", abs(a_kal - a_roh) < 1e-9,
          f"{a_roh:.6f} gegen {a_kal:.6f}")
    check("die mittlere Vorhersage trifft die Haeufigkeit",
          abs(pc.mean() - y.mean()) < 0.01,
          f"{pc.mean():.4f} gegen {y.mean():.4f}")
    check("der Kalibrierfehler faellt", ece(y, pc) < ece(y, p),
          f"ECE {ece(y, p):.4f} -> {ece(y, pc):.4f}")
    check("der Brier-Wert faellt", brier(y, pc) < brier(y, p),
          f"{brier(y, p):.4f} -> {brier(y, pc):.4f}")

    # Ein bereits kalibrierter Satz darf nicht verbogen werden: dann muss der
    # Fit ungefaehr die Identitaet finden.
    p_gut = 1 / (1 + np.exp(-(rng.normal(0, 1.2, n))))
    y_gut = (rng.random(n) < p_gut).astype(float)
    a2, b2 = platt_fit(p_gut, y_gut)
    check("ein kalibrierter Satz ergibt ungefaehr die Identitaet",
          abs(a2 - 1) < 0.12 and abs(b2) < 0.12, f"a {a2:.4f}, b {b2:+.4f}")

    # Zwei Umsetzungen derselben Kurve. rsna_holdout rechnet die Kurve beim
    # Vorhersagen noch einmal; laufen die beiden auseinander, rechnet die
    # Auswertung anders als der Lauf.
    try:
        from rsna_holdout import platt_apply as platt_apply_hold
        gleich = np.allclose(platt_apply_fit(p, a, b), platt_apply_hold(p, a, b),
                             rtol=0, atol=0)
        check("rsna_platt und rsna_holdout wenden dieselbe Kurve bitgleich an",
              bool(gleich))
    except ImportError as e:                      # torch fehlt in der Umgebung
        check("rsna_holdout ist importierbar", False, str(e))


def test_youden():
    print("\nDIE SCHWELLE")
    # Ein Fall mit bekannter Antwort: perfekt trennbar bei 0,5.
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr, sens, spez = youden(y, p)
    check("perfekte Trennung wird gefunden", sens == 1.0 and spez == 1.0,
          f"Schwelle {thr:.2f}, Sens {sens:.2f}, Spez {spez:.2f}")
    check("die Schwelle sitzt auf dem kleinsten positiven Wert", thr == 0.7,
          f"{thr}")

    # Und ein Fall, in dem die beste Trennung NICHT bei 0,5 liegt.
    y2 = np.array([0, 0, 1, 0, 1, 1], dtype=float)
    p2 = np.array([0.05, 0.10, 0.12, 0.14, 0.20, 0.30])
    thr2, s2, sp2 = youden(y2, p2)
    check("eine Schwelle weit unter 0,5 wird gefunden", thr2 == 0.12,
          f"{thr2}, Sens {s2:.2f}, Spez {sp2:.2f}")


def test_geschichtete_auc():
    print("\nGESCHICHTET RECHNEN, und warum es sein muss")
    # Innerhalb jeder Aufnahmeart ist der Wert reiner Zufall, aber AP hat mehr
    # Kranke UND hoehere Werte. Die rohe AUC belohnt das, die geschichtete
    # nicht. Genau dieser Unterschied ist das Thema des Projekts.
    rng = np.random.default_rng(1)
    vp = np.array(["AP"] * 1000 + ["PA"] * 1000)
    y = np.r_[(rng.random(1000) < 0.40), (rng.random(1000) < 0.10)].astype(float)
    s = np.r_[rng.normal(1.0, 1, 1000), rng.normal(0.0, 1, 1000)]
    roh = rank_auc(s, y)
    strat = stratified_auc(s, y, vp)
    check("die rohe AUC liegt deutlich ueber dem Muenzwurf", roh > 0.60,
          f"{roh:.4f}")
    check("die geschichtete AUC entlarvt das als Muenzwurf", abs(strat - 0.5) < 0.03,
          f"{strat:.4f}")

    # Zwei Umsetzungen derselben Kennzahl.
    check("rsna_platt und die Auswertung rechnen dieselbe AUC",
          abs(rank_auc_platt(s, y) - roh) < 1e-12)


def test_sperre():
    print("\nDIE SPERRE AUF DEM HOLDOUT")
    try:
        from rsna_holdout import SOLL_HOLDOUT_N, SPERRE
    except ImportError as e:
        check("rsna_holdout ist importierbar", False, str(e))
        return
    check("der Lauf und die Auswertung meinen dieselbe Anzahl",
          SOLL_HOLDOUT_N == SOLL_N, f"{SOLL_HOLDOUT_N} gegen {SOLL_N}")
    src = (Path("rsna") / "befunde" / "rsna_phase10_auswertung.py").read_text(
        encoding="utf-8")
    check("die Auswertung sucht genau die Sperrdatei, die der Lauf schreibt",
          SPERRE in src, SPERRE)
    check("die Auswertung bricht ohne Sperrdatei ab",
          "sperre.is_file()" in src and "abbruch" in src)
    check("die Auswertung liest, ob es der erste Blick war",
          "erster_blick" in src)

    # Und die Datei selbst, falls sie schon existiert: dann muss dort stehen,
    # ob es der erste Blick war. Ein Lauf ohne dieses Feld waere aus einer
    # aelteren Fassung und darf nicht stillschweigend als erster gelten.
    pfad = Path("predictions_holdout") / SPERRE
    if pfad.is_file():
        lock = json.loads(pfad.read_text(encoding="utf-8"))
        check("die vorhandene Sperrdatei nennt den ersten Blick",
              "erster_blick" in lock,
              f"erster_blick={lock.get('erster_blick')}")
    else:
        print("  --    noch keine Sperrdatei, der Holdout ist unberuehrt")


if __name__ == "__main__":
    test_tor_a()
    test_konstanten()
    test_platt()
    test_youden()
    test_geschichtete_auc()
    test_sperre()
    print("\n" + ("ALLE TESTS BESTANDEN" if not FAILED
                  else f"{len(FAILED)} FEHLGESCHLAGEN: {FAILED}"))
    sys.exit(1 if FAILED else 0)
