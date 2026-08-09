"""Tests fuer Phase 5b Teil 2, die Kaesten aus dem Kopffeld.

DIE STILLEN FEHLER, GEGEN DIE DIESE DATEI EXISTIERT
---------------------------------------------------
Hier wird eine Zahl erzeugt, die spaeter neben Wettbewerbsergebnissen stehen
soll. Eine selbstgebaute Metrik, die um ein paar Prozent danebenliegt, faellt
niemandem auf und macht genau diesen Vergleich falsch. Deshalb steht am Anfang
kein eigener Test, sondern ein Abgleich mit der Referenzumsetzung des
Wettbewerbs, Zeile fuer Zeile nachgebaut.

  * DIE METRIK WEICHT VON DER DES WETTBEWERBS AB. Gegen die Referenz auf 400
    Zufallsfaellen geprueft, verlangt wird EXAKTE Gleichheit, nicht Naehe.
  * DIE VIER SONDERFAELLE. Kein Kasten und keine Vorhersage wird NICHT
    gewertet, kein Kasten mit Vorhersage zaehlt als 0, Kasten ohne Vorhersage
    zaehlt als 0, perfekter Treffer ist 1. Der erste ist der, den man beim
    Nachbauen am ehesten falsch macht, und er verschiebt das Ergebnis in die
    schmeichelhafte Richtung.
  * DIE ACHT IoU-SCHWELLEN WIRKEN WIRKLICH. Bei gleicher Breite und gleichem
    Ursprung ist die Ueberlappung exakt einstellbar, also ist auch das
    erwartete Ergebnis exakt bekannt.
  * DIE SORTIERUNG NACH KONFIDENZ TUT ETWAS. Ohne Gegentest waere nicht
    belegt, dass die Reihenfolge ueberhaupt eine Rolle spielt.
  * KACHELN ZU KASTEN. Der Kasten einer Region muss auf den Pixel stimmen,
    sonst sind alle Ueberlappungen leise verschoben.
  * DAS KLASSIFIKATORTOR. Ein unterdruecktes Bild MIT Kasten muss 0 zaehlen,
    eines OHNE Kasten muss aus dem Mittel fallen. Wer beide gleich behandelt,
    bekommt ein besseres Ergebnis als er verdient.

  python tests\\test_rsna_detektion.py
"""

from __future__ import annotations

import _repo_path  # noqa: F401

import sys

import numpy as np

from rsna_phase5b_detektion import (IOU_THR, iou, map_iou, regionen,
                                    score_mit_tor)

FAILED = []


def check(name: str, cond: bool, info: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FAILED.append(name)


def referenz(bt, bp, sc, thresholds=IOU_THR):
    """Die Umsetzung aus dem Wettbewerb, absichtlich woertlich nachgebaut.

    Nicht importiert und nicht aufgeraeumt: eine Abweichung soll eine echte
    Meinungsverschiedenheit sein und kein geteilter Fehler.
    """
    if len(bt) == 0 and len(bp) == 0:
        return None
    if len(bp):
        bp = bp[np.argsort(sc)[::-1], :]
    total = 0
    for t in thresholds:
        matched = set()
        tp, fn = 0, 0
        for b in bt:
            m = False
            for j, p in enumerate(bp):
                if iou(b, p) >= t and not m and j not in matched:
                    m = True
                    tp += 1
                    matched.add(j)
            if not m:
                fn += 1
        fp = len(bp) - len(matched)
        total += tp / (tp + fn + fp)
    return total / len(thresholds)


def zufallskasten(rng, n):
    if not n:
        return np.zeros((0, 4))
    return np.column_stack([rng.uniform(0, 700, n), rng.uniform(0, 700, n),
                            rng.uniform(80, 400, n), rng.uniform(80, 400, n)])


def test_gegen_die_referenz() -> None:
    print("\ntest_gegen_die_referenz")
    rng = np.random.default_rng(0)
    worst, n, none_ok = 0.0, 0, True
    for _ in range(400):
        bt = zufallskasten(rng, int(rng.integers(0, 4)))
        bp = zufallskasten(rng, int(rng.integers(0, 4)))
        sc = rng.random(len(bp))
        a = map_iou(bt, bp, sc)
        b = referenz(bt.copy(), bp.copy(), sc.copy())
        if a is None or b is None:
            none_ok = none_ok and (a is None) and (b is None)
        else:
            worst = max(worst, abs(a - b))
            n += 1
    check("400 Zufallsfaelle stimmen EXAKT mit der Referenz ueberein",
          worst == 0.0 and none_ok,
          f"{n} mit Wert verglichen, groesste Abweichung {worst:.1e}")


def test_die_vier_sonderfaelle() -> None:
    print("\ntest_die_vier_sonderfaelle")
    leer = np.zeros((0, 4))
    b = np.array([[100., 100., 200., 200.]])
    check("kein Kasten und keine Vorhersage wird NICHT gewertet",
          map_iou(leer, leer, np.zeros(0)) is None)
    check("kein Kasten, aber eine Vorhersage: 0 und gewertet",
          map_iou(leer, b, np.ones(1)) == 0.0)
    check("Kasten, aber keine Vorhersage: 0",
          map_iou(b, leer, np.zeros(0)) == 0.0)
    check("perfekter Treffer: 1", map_iou(b, b.copy(), np.ones(1)) == 1.0)


def test_die_acht_schwellen() -> None:
    """Bei gleicher Breite und gleichem Ursprung ist IoU = h / 200, also exakt
    einstellbar, und damit ist auch das Ergebnis exakt vorhersagbar."""
    print("\ntest_die_acht_schwellen")
    b = np.array([[100., 100., 200., 200.]])
    for h in (70., 110., 160., 200.):
        p = np.array([[100., 100., 200., h]])
        treffer = sum(1 for t in IOU_THR if iou(b[0], p[0]) >= t)
        s = map_iou(b, p, np.ones(1))
        check(f"Ueberlappung {iou(b[0], p[0]):.2f} zaehlt auf {treffer} von 8",
              abs(s - treffer / 8) < 1e-12, f"gerechnet {s:.4f}")


def test_sortierung_wirkt() -> None:
    """Die Konstellation, in der die Reihenfolge das Ergebnis aendert.

    Sie muss GEBAUT werden. Zufaellige Kaesten erzeugen sie praktisch nie: es
    braucht eine Vorhersage, die zu ZWEI Annotationen passt, und eine zweite,
    die nur zu einer passt. Ein erster Anlauf mit 400 Zufallsfaellen fand null
    solche Faelle, was den Test wertlos gemacht haette, ohne dass es auffaellt.

    Aufbau: zwei ueberlappende Annotationen, P1 passt zu beiden (IoU 0.6), P2
    ist die exakte Kopie der ersten (IoU 1.0 dort, 0.33 zur zweiten und damit
    unter jeder Schwelle). Wer P2 zuerst zuordnet, bekommt beide Annotationen
    getroffen. Wer P1 zuerst nimmt, verbraucht ihn an der ersten und laesst die
    zweite ohne Partner.
    """
    print("\ntest_sortierung_wirkt")
    bt = np.array([[0., 0., 200., 200.], [0., 100., 200., 200.]])
    p1 = [0., 50., 200., 200.]
    p2 = [0., 0., 200., 200.]
    bp = np.array([p1, p2])
    check(f"P1 passt zu beiden Annotationen "
          f"({iou(bt[0], p1):.2f} und {iou(bt[1], p1):.2f})",
          iou(bt[0], p1) >= 0.4 and iou(bt[1], p1) >= 0.4)
    check(f"P2 passt nur zur ersten "
          f"({iou(bt[0], p2):.2f} und {iou(bt[1], p2):.2f})",
          iou(bt[0], p2) >= 0.4 and iou(bt[1], p2) < 0.4)
    hoch = map_iou(bt, bp, np.array([0.1, 0.9]))     # P2 zuerst
    tief = map_iou(bt, bp, np.array([0.9, 0.1]))     # P1 zuerst
    check("die Reihenfolge aendert das Ergebnis", abs(hoch - tief) > 1e-9,
          f"P2 zuerst {hoch:.4f}, P1 zuerst {tief:.4f}")
    check("die Sortierung waehlt die bessere Reihenfolge", hoch > tief)


def test_kacheln_zu_kasten() -> None:
    print("\ntest_kacheln_zu_kasten")
    f = np.zeros((14, 14), np.float32)
    f[2:5, 3:6] = 0.9          # ein Block aus 3 x 3 Kacheln
    f[10, 11] = 0.95           # eine einzelne Kachel
    r = regionen(f, 0.5)
    check("zwei getrennte Regionen", len(r) == 2, f"{len(r)}")
    gross = [x for x in r if x[2] == 9][0]
    klein = [x for x in r if x[2] == 1][0]
    s = 1024 / 14
    check("der Kasten des Blocks stimmt auf den Pixel",
          bool(np.allclose(gross[0], (3 * s, 2 * s, 3 * s, 3 * s))),
          str(np.round(gross[0], 1)))
    check("die Konfidenz ist das Maximum der Region",
          bool(np.isclose(gross[1], 0.9, atol=1e-6)
               and np.isclose(klein[1], 0.95, atol=1e-6)),
          f"{gross[1]:.6f} und {klein[1]:.6f}")
    check("min_tiles entfernt die einzelne Kachel",
          len([x for x in r if x[2] >= 2]) == 1)
    check("ueber dem Maximum des Feldes gibt es keine Region",
          regionen(f, 0.99) == [])
    # Diagonal beruehrende Kacheln sind ZWEI Regionen, nicht eine. Mit
    # Achter-Nachbarschaft waeren es eine, und der Kasten waere doppelt so
    # gross wie das, was das Modell gezeigt hat.
    g = np.zeros((14, 14), np.float32)
    g[3, 3] = g[4, 4] = 0.9
    check("diagonal beruehrende Kacheln bleiben getrennt", len(regionen(g, 0.5)) == 2)


def test_das_klassifikatortor() -> None:
    print("\ntest_das_klassifikatortor")
    m = np.array([0.8, 0.6, np.nan, 0.0])
    g = np.array([True, True, False, False])
    n = np.array([1, 1, 0, 2])
    p = np.array([0.9, 0.1, 0.9, 0.1])
    s0, n0, _ = score_mit_tor(m, g, n, p, -1.0)
    check("ohne Tor zaehlt der Mittelwert der gewerteten Bilder",
          abs(s0 - (0.8 + 0.6 + 0.0) / 3) < 1e-12, f"{s0:.4f} ueber {n0} Bilder")
    s1, n1, _ = score_mit_tor(m, g, n, p, 0.5)
    check("unterdrueckt MIT Kasten zaehlt 0, unterdrueckt OHNE Kasten faellt raus",
          abs(s1 - (0.8 + 0.0) / 2) < 1e-12 and n1 == 2,
          f"{s1:.4f} ueber {n1} Bilder")
    s2, _, anteil = score_mit_tor(m, g, n, p, 2.0)
    check("ein Tor, das alles unterdrueckt, laesst nur die Kasten-Bilder mit 0",
          abs(s2 - 0.0) < 1e-12 and anteil == 0.0)


if __name__ == "__main__":
    test_gegen_die_referenz()
    test_die_vier_sonderfaelle()
    test_die_acht_schwellen()
    test_sortierung_wirkt()
    test_kacheln_zu_kasten()
    test_das_klassifikatortor()
    print("\n" + ("ALLE TESTS BESTANDEN" if not FAILED
                  else f"{len(FAILED)} FEHLGESCHLAGEN: {FAILED}"))
    sys.exit(1 if FAILED else 0)
