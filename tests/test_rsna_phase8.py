"""Rauchtests fuer das Urteil der Phase 8.

DIE STILLEN FEHLER, GEGEN DIE DIESE DATEI EXISTIERT
---------------------------------------------------
Das Auswertungsskript der Phase 8 faellt ein Urteil, das nach dem Lauf nicht
mehr verhandelt wird. Ein Vorzeichenfehler in einer der drei Urteilsfunktionen
faellt dann niemandem auf, weil es keine zweite Meinung gibt: das Ergebnis
STEHT dann einfach da, und es sieht genauso aus wie ein richtiges.

Deshalb werden hier die Aeste einzeln an Faellen geprueft, deren Antwort
feststeht, samt der Randfaelle GENAU AUF der Schwelle. Das ist dieselbe Regel
wie in `fehler_messgeraet_nach_dem_ergebnis_wechseln`: ein Messgeraet zuerst an
einem Fall pruefen, dessen Antwort man schon kennt.

Vier Dinge werden geprueft:

  * Tor A besteht nur, wenn BEIDES gilt, gesichert gestiegen UND mindestens
    +0,008. Ein gesicherter Anstieg von +0,005 ist kein bestandenes Tor, und
    genau das ist der wahrscheinlichste Ausgang dieser Phase.
  * Tor C unterscheidet vier Aeste, und die Grauzonenschwelle liegt bei
    -0,015. Der Fall GENAU auf der Schwelle gehoert in die Grauzone.
  * Der Riegel auf Tor C entscheidet am UNTEREN Ende und an der Marge 0,01.
  * Die Konstanten im Skript sind die der Vorfestlegung. Eine Zahl, die im
    Skript still von der Vorfestlegung abweicht, waere ein nachtraeglich
    geaendertes Tor, und dagegen hilft keine Regel im Gedaechtnis, sondern nur
    ein Test.

Kein Training, keine GPU, keine Daten: geprueft wird die Verdrahtung des
Urteils.

  python tests\\test_rsna_phase8.py
"""

from __future__ import annotations

import sys

import _repo_path  # noqa: F401

from rsna_phase8_auswertung import (ANKER_A, ANKER_C, GRAUZONE, MARGE_A,
                                    MINDEST_A, NIVEAU, urteil_a,
                                    urteil_a_nichtunterlegen, urteil_c)

FAILED = []


def check(name: str, cond: bool, info: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FAILED.append(name)


def r(mean: float, lo: float, hi: float) -> dict:
    """Ein Ergebnis, wie `gepaart` es liefert. Nur die drei Zahlen, die die
    Urteilsfunktionen ueberhaupt lesen."""
    return {"mean": mean, "lo": lo, "hi": hi, "sd": 0.0, "n": 5, "je_fold": []}


def test_tor_a():
    print("\nTOR A: A muss STEIGEN, um mindestens", MINDEST_A)
    faelle = [
        ("klar gestiegen, ueber der Latte", r(+0.012, +0.006, +0.018),
         "BESTANDEN"),
        ("GENAU auf der Latte", r(+0.008, +0.002, +0.014), "BESTANDEN"),
        ("gesichert gestiegen, aber unter der Latte",
         r(+0.005, +0.001, +0.009), "DURCHGEFALLEN"),
        ("ueber der Latte, aber Null im Intervall",
         r(+0.012, -0.002, +0.026), "DURCHGEFALLEN"),
        ("gesichert gefallen", r(-0.015, -0.022, -0.008), "DURCHGEFALLEN"),
    ]
    for name, x, soll in faelle:
        ist = urteil_a(x)[0]
        check(name, ist == soll, f"{ist} (erwartet {soll})")
    # Der Fall, der die Phase am ehesten trifft, noch einmal ausdruecklich:
    # Phase 7 hat +0,0084 gemessen, das ist der groesste je gemessene
    # A-Gewinn aus einer Bildaenderung. Ein Tor bei 0,01 haette ihn NICHT
    # durchgelassen, ein Tor bei 0,008 laesst ihn durch. Genau darum wurde die
    # Latte vor dem Lauf gesenkt.
    check("der Phase-7-Gewinn +0,0084 wuerde dieses Tor oeffnen",
          urteil_a(r(+0.0084, +0.0028, +0.0139))[0] == "BESTANDEN")
    check("bei der alten Latte 0,01 wuerde er es nicht",
          0.0084 < 0.01)


def test_tor_c():
    print("\nTOR C: C muss FALLEN, Grauzone ab", GRAUZONE)
    faelle = [
        ("gesichert gefallen", r(-0.030, -0.050, -0.010), "BESTANDEN"),
        ("Null drin, GENAU auf der Grauzonenschwelle",
         r(-0.015, -0.035, +0.005), "GRAUZONE"),
        ("Null drin, knapp ueber der Schwelle",
         r(-0.014, -0.034, +0.006), "DURCHGEFALLEN"),
        ("Null drin, deutlich darueber", r(-0.002, -0.026, +0.022),
         "DURCHGEFALLEN"),
        ("gesichert gestiegen", r(+0.020, +0.005, +0.035), "DURCHGEFALLEN"),
    ]
    for name, x, soll in faelle:
        ist = urteil_c(x)[0]
        check(name, ist == soll, f"{ist} (erwartet {soll})")
    # Die Vorgeschichte, an der die Teststaerke haengt: der 320-px-Punktwert
    # war -0,0327, die Halbbreite auf C liegt bei rund 0,025. Traefe die volle
    # Wirkung zu, ginge das Tor auf; bei der halben Wirkung nicht.
    check("die volle vermutete Wirkung (-0,0327) oeffnet das Tor",
          urteil_c(r(-0.0327, -0.0577, -0.0077))[0] == "BESTANDEN")
    check("die halbe Wirkung (-0,016) landet in der Grauzone",
          urteil_c(r(-0.016, -0.041, +0.009))[0] == "GRAUZONE")


def test_riegel():
    print("\nRIEGEL auf Tor C: A nicht unterlegen, Marge", MARGE_A)
    faelle = [
        ("A leicht gefallen, innerhalb der Marge",
         r(-0.005, -0.009, -0.001), "BESTANDEN"),
        ("unteres Ende GENAU auf der Marge", r(-0.008, -0.010, -0.006),
         "DURCHGEFALLEN"),
        ("Marge gerissen", r(-0.008, -0.014, -0.002), "DURCHGEFALLEN"),
        ("A gestiegen", r(+0.010, +0.004, +0.016), "BESTANDEN"),
    ]
    for name, x, soll in faelle:
        ist = urteil_a_nichtunterlegen(x)[0]
        check(name, ist == soll, f"{ist} (erwartet {soll})")
    # Der Gegentest zur Reihenfolge: ein bestandenes Tor A kann den Riegel
    # nicht reissen. Waere das moeglich, widerspraeche sich das Urteil selbst.
    check("ein bestandenes Tor A oeffnet immer auch den Riegel",
          all(urteil_a_nichtunterlegen(x)[0] == "BESTANDEN"
              for x in (r(+0.008, +0.001, +0.015), r(+0.020, +0.010, +0.030),
                        r(+0.009, +0.0001, +0.018))))


def test_konstanten():
    print("\nDIE KONSTANTEN gegen erklaerungen\\25_phase8_vorfestlegung.md")
    soll = {"ANKER_A": (ANKER_A, 0.8368), "ANKER_C": (ANKER_C, 0.7467),
            "MINDEST_A": (MINDEST_A, 0.008), "MARGE_A": (MARGE_A, 0.01),
            "GRAUZONE": (GRAUZONE, -0.015), "NIVEAU": (NIVEAU, 0.90)}
    for name, (ist, s) in soll.items():
        check(f"{name} = {s}", abs(ist - s) < 1e-12, f"gelesen {ist}")


if __name__ == "__main__":
    test_tor_a()
    test_tor_c()
    test_riegel()
    test_konstanten()
    print("\n" + ("ALLE TESTS BESTANDEN" if not FAILED
                  else f"{len(FAILED)} FEHLGESCHLAGEN: {FAILED}"))
    sys.exit(1 if FAILED else 0)
