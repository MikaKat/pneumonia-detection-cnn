"""Rauchtests fuer das Urteil UND die Verkabelung der Phase 9.

DIE STILLEN FEHLER, GEGEN DIE DIESE DATEI EXISTIERT
---------------------------------------------------
Zwei Sorten, und die zweite ist neu.

Die erste ist die aus Phase 8: das Auswertungsskript faellt ein Urteil, das
nach dem Lauf nicht mehr verhandelt wird, und ein Vorzeichenfehler in einer der
Urteilsfunktionen faellt niemandem auf, weil es keine zweite Meinung gibt.
Dagegen stehen unten Faelle, deren Antwort feststeht, samt der Randfaelle GENAU
AUF der Schwelle.

Die zweite ist der Anlass dieser Phase. Bis zum 09.08.2026 sassen `brightness`
und `contrast` fest verdrahtet mit 0,15 im Konstruktor von `TrainTransform`,
waehrend Verschiebung, Skalierung und Rotation laengst Argumente waren. Genau
diese Gestalt hatte `--balance-strength`, als es gelesen und dann nicht
weitergereicht wurde. Ein Arm, der den Schalter setzt und trotzdem mit 0,15
laeuft, sieht wie ein sauberes Nullergebnis aus, und nichts in der Ausgabe
widerspricht. Deshalb wird hier nicht geprueft, ob der Schalter existiert,
sondern ob er ANKOMMT, und zwar an dem Objekt, das der DataLoader wirklich
haelt.

Vier Dinge werden geprueft:

  * Das Tor auf C unterscheidet vier Aeste, Grauzonenschwelle -0,015, und der
    Fall GENAU auf der Schwelle gehoert in die Grauzone.
  * Der Riegel entscheidet am UNTEREN Ende und an der Marge 0,01.
  * Die Konstanten im Auswertungsskript sind die der Vorfestlegung.
  * Die photometrische Staerke kommt in der Transformation an, die Vorgabe
    bleibt bitgleich zu allem, was vor dem 09.08. lief, und das Messen selbst
    verbraucht keinen Zufall.

Keine GPU, keine Daten, kein Training.

  python tests\\test_rsna_phase9.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

import _repo_path  # noqa: F401

from rsna_phase9_auswertung import (ANKER_A, ANKER_C, ARM_PHOTO, BEZUG_PHOTO,
                                    GRAUZONE, MARGE_A, NIVEAU, SOLL_SIZE,
                                    urteil_a_nichtunterlegen, urteil_c)
from rsna_train import TrainTransform, gemessene_jitter_staerke

FAILED = []


def check(name: str, cond: bool, info: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FAILED.append(name)


def r(mean: float, lo: float, hi: float) -> dict:
    """Ein Ergebnis, wie `gepaart` es liefert. Nur die drei Zahlen, die die
    Urteilsfunktionen ueberhaupt lesen."""
    return {"mean": mean, "lo": lo, "hi": hi, "sd": 0.0, "n": 5, "je_fold": []}


def test_tor_c():
    print("\nPRIMAER: C muss FALLEN, Grauzone ab", GRAUZONE)
    faelle = [
        ("gesichert gefallen", r(-0.030, -0.050, -0.010), "BESTANDEN"),
        # Die Vorfestlegung sagt "das obere Ende UNTER null", nicht "hoechstens
        # null". Ein Intervall, das die Null beruehrt, ist also nicht
        # bestanden; es faellt hier in die Grauzone, weil der Punktwert unter
        # -0,015 liegt. Dieselbe Strenge steht in Phase 8, damit die Arme
        # vergleichbar bleiben. Der Fall stand zuerst mit der falschen
        # Erwartung hier und ist beim ersten Lauf aufgefallen.
        ("oberes Ende GENAU auf null zaehlt NICHT als gefallen",
         r(-0.030, -0.060, 0.0), "GRAUZONE"),
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
    # Die Groessenordnungen, an denen die Phase haengt. Halbbreite rund 0,023,
    # also ist ein Punktwert von -0,03 gerade eben gesichert und einer von
    # -0,02 gerade eben nicht.
    check("ein Effekt wie die volle Entkopplung (-0,0554) oeffnet das Tor",
          urteil_c(r(-0.0554, -0.0784, -0.0324))[0] == "BESTANDEN")
    check("ein Effekt wie Phase 6 (-0,0052) tut es nicht",
          urteil_c(r(-0.0052, -0.0282, +0.0178))[0] == "DURCHGEFALLEN")


def test_riegel():
    print("\nRIEGEL: A nicht unterlegen, Marge", MARGE_A)
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
    # Der Fall, der diese Phase am ehesten trifft: ein starker Jitter kostet
    # Trennschaerfe. Bei einem Verlust von 0,012 muss der Riegel greifen, sonst
    # waere ein C-Rueckgang nicht mehr von "schlechterer Trenner" zu
    # unterscheiden.
    check("ein A-Verlust von 0,012 reisst den Riegel",
          urteil_a_nichtunterlegen(r(-0.012, -0.018, -0.006))[0]
          == "DURCHGEFALLEN")


def test_konstanten():
    print("\nDIE KONSTANTEN gegen erklaerungen\\27_phase9_photometrisch.md")
    soll = {"ANKER_A": (ANKER_A, 0.8368), "ANKER_C": (ANKER_C, 0.7467),
            "MARGE_A": (MARGE_A, 0.01), "GRAUZONE": (GRAUZONE, -0.015),
            "NIVEAU": (NIVEAU, 0.90), "ARM_PHOTO": (ARM_PHOTO, 0.60),
            "BEZUG_PHOTO": (BEZUG_PHOTO, 0.15),
            "SOLL_SIZE": (SOLL_SIZE, 224)}
    for name, (ist, s) in soll.items():
        check(f"{name} = {s}", abs(ist - s) < 1e-12, f"gelesen {ist}")


def test_verkabelung():
    print("\nDIE VERKABELUNG: kommt die Staerke in der Transformation an?")
    for b in (BEZUG_PHOTO, 0.40, ARM_PHOTO):
        tf = TrainTransform(224, 0.03, (0.93, 1.07), 7.0, b, b)
        mb, mc = gemessene_jitter_staerke(tf)
        schranke = max(0.25 * b, 0.03)
        check(f"Staerke {b:.2f} wird auch gezogen",
              abs(mb - b) <= schranke and abs(mc - b) <= schranke,
              f"gemessen {mb:.4f} / {mc:.4f}, Schranke {schranke:.3f}")

    # DER GEGENTEST, und er ist der eigentliche Punkt dieser Datei: der
    # Schalter steht auf 0,60, die Transformation zieht aber 0,15. Genau so
    # sieht ein nicht verkabelter Schalter aus.
    tf = TrainTransform(224, 0.03, (0.93, 1.07), 7.0, ARM_PHOTO, ARM_PHOTO)
    tf.jitter = __import__("torchvision").transforms.ColorJitter(
        brightness=BEZUG_PHOTO, contrast=BEZUG_PHOTO)
    mb, mc = gemessene_jitter_staerke(tf)
    schranke = max(0.25 * ARM_PHOTO, 0.03)
    check("ein NICHT verkabelter Schalter faellt durch",
          abs(mb - ARM_PHOTO) > schranke,
          f"Schalter {ARM_PHOTO}, gezogen {mb:.4f}")

    # Die Vorgabe muss bitgleich zu allem sein, was vor dem 09.08. lief. Sonst
    # waere der Bezugsarm kein gueltiger Partner mehr, und mit ihm haengen
    # Phase 6, 7 und 8 am selben Anker.
    a = np.arange(224 * 224, dtype=np.int64) % 251
    from PIL import Image
    img = Image.fromarray(a.reshape(224, 224).astype(np.uint8), mode="L")
    torch.manual_seed(3)
    alt = TrainTransform(224)(img)
    torch.manual_seed(3)
    neu = TrainTransform(224, 0.03, (0.93, 1.07), 7.0, 0.15, 0.15)(img)
    check("die Vorgabe ist bitgleich zur alten Verdrahtung",
          bool(torch.equal(alt, neu)))

    # Das Messen darf dem Training keinen Zufall wegnehmen. Sonst waere ein
    # Lauf mit Messung nicht mehr derselbe wie einer ohne, und die
    # Wiederholbarkeit haenge an einer Pruefung.
    torch.manual_seed(7)
    vorher = torch.randn(4)
    torch.manual_seed(7)
    gemessene_jitter_staerke(TrainTransform(224, 0.03, (0.93, 1.07), 7.0,
                                            ARM_PHOTO, ARM_PHOTO))
    nachher = torch.randn(4)
    check("das Messen verbraucht keinen Zufall",
          bool(torch.equal(vorher, nachher)))

    # Und die Grenze des Messgeraets, ausdruecklich: ueber 0,9 klemmt
    # torchvision das untere Ende des Faktorbereichs bei 0 fest, die Ziehung
    # ist dann nicht mehr gleichverteilt auf [1-b, 1+b], und die Rueckrechnung
    # laese eine Staerke ab, die nicht die wirksame ist.
    try:
        gemessene_jitter_staerke(TrainTransform(224, 0.03, (0.93, 1.07), 7.0,
                                                0.95, 0.95))
        check("eine Staerke ueber 0,9 wird abgelehnt", False, "kein Abbruch")
    except SystemExit as e:
        check("eine Staerke ueber 0,9 wird abgelehnt", "0.9" in str(e))


if __name__ == "__main__":
    test_tor_c()
    test_riegel()
    test_konstanten()
    test_verkabelung()
    print("\n" + ("ALLE TESTS BESTANDEN" if not FAILED
                  else f"{len(FAILED)} FEHLGESCHLAGEN: {FAILED}"))
    sys.exit(1 if FAILED else 0)
