"""Rauchtests fuer den zweiten Kopf, Phase 5.

DIE STILLEN FEHLER, GEGEN DIE DIESE DATEI EXISTIERT
---------------------------------------------------
Die Roadmap verlangt Tests auf die WIRKUNG, nicht auf die Ausgabe: dass ein
Skript ohne Fehler durchlaeuft und Dateien schreibt, beweist nichts. Der
teuerste Fehler dieses Projekts war ein Lauf, der etwas anderes tat, als er
ankuendigte.

Sechs Dinge werden geprueft, und das erste ist das wichtige:

  * DIE KASTENFALLE. Bild und Kastenmaske muessen dieselbe Bewegung machen.
    Geprueft an Schwerpunkten: ein Bild, das nichts enthaelt ausser einem
    hellen Rechteck, muss nach der Transformation seinen Schwerpunkt dort
    haben, wo die Maske ihren hat. Dazu der GEGENTEST, ohne den der erste
    nichts wert waere: dieselbe Messung mit zwei getrennten Ziehungen muss
    FEHLSCHLAGEN. Ein Test, der auch die kaputte Fassung durchwinkt, prueft
    nichts.
  * Mit und ohne Maske muss derselbe Zufallsstrom laufen, sonst sehen die
    beiden Arme verschiedene Bilder und der gepaarte Vergleich misst zwei
    Dinge.
  * Das weiche Ziel traegt wirklich den Flaechenanteil und ist wirklich weich.
  * `pos_weight` je Kachel unterscheidet sich zwischen den beiden
    Negativen-Varianten, und zwar um genau den Faktor, um den die Bedeckung
    faellt.
  * Der Kopf gibt 14 mal 14 aus, unabhaengig von der Bildgroesse. Das ist der
    halbe Sinn der Phase.
  * Der Klassifikationsweg des zweikoepfigen Modells rechnet BITGLEICH
    dasselbe wie das einkoepfige.
  * LAMBDA DARF NICHT AUS EINEM STAPEL OHNE KASTEN KOMMEN. Das ist der Fehler,
    der am 05.08.2026 einen Fold gekostet hat: der erste Stapel enthielt kein
    annotiertes Bild, der Kopfverlust war exakt null, der Abfang bei 1e-8 machte
    aus dem Verhaeltnis 7,5e7, und der Rumpf trainierte danach fast nur noch auf
    die Kaesten. Der Lauf sah bis zum Ende normal aus. Auch hier gehoert der
    GEGENTEST dazu: die alte Regel muss durchfallen.

Kein Training, keine GPU, keine ImageNet-Gewichte: das Modell wird mit
`pretrained=False` gebaut, weil hier nur die Verdrahtung geprueft wird.

  python tests\\test_rsna_kopf.py
"""

from __future__ import annotations

import _repo_path  # noqa: F401

import sys

import numpy as np

FAILED = []


def check(name: str, cond: bool, info: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FAILED.append(name)


try:
    import torch
    from PIL import Image
    from torchvision import transforms as T
    from torchvision.transforms import functional as TF
except ImportError as e:                       # pragma: no cover
    print(f"torch oder PIL fehlt ({e}), diese Tests brauchen beides.")
    raise SystemExit(2)

from rsna_train import (HEAD_GRID, LAMBDA_MAX, LAMBDA_MIN, TrainTransform,
                        TwoHeadNet, batch_can_set_lambda, box_mask_pil,
                        loc_loss, tile_coverage, tile_pos_weight)

SIZE = 224
BOX = [(300.0, 250.0, 260.0, 300.0)]      # ein Kasten im 1024er DICOM-Raster


def centroid(a) -> tuple[float, float]:
    """Schwerpunkt eines Feldes, in Bildpunkten."""
    a = np.asarray(a, dtype=np.float64)
    a = np.clip(a - a.min(), 0, None)
    tot = a.sum()
    if tot <= 0:
        raise ValueError("leeres Feld, der Schwerpunkt waere undefiniert")
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    return float((a * yy).sum() / tot), float((a * xx).sum() / tot)


def synthetic_pair():
    """Ein Bild, das NUR den Kasten zeigt, und die zugehoerige Maske.

    Das ist der Trick, der den Test einfach macht: enthaelt das Bild ausser dem
    Kasten nichts, ist sein Schwerpunkt der Schwerpunkt des Kastens. Bleiben
    beide nach der Transformation zusammen, hat die Maske dieselbe Bewegung
    gemacht wie das Bild.

    Das Bild wird in der ORIGINALGROESSE 512 gebaut, damit die
    Groessenaenderung mit im Test steckt und nicht nur die Affinbewegung.
    """
    m512 = box_mask_pil(BOX, 512)
    return Image.fromarray(np.asarray(m512, np.uint8), mode="L"), box_mask_pil(BOX, SIZE)


# --------------------------------------------------------------------------

def test_kastenfalle() -> None:
    """Bild und Maske duerfen nicht auseinanderlaufen."""
    print("\ntest_kastenfalle")
    img, mask = synthetic_pair()
    tf = TrainTransform(SIZE)
    worst = 0.0
    for seed in range(16):
        torch.manual_seed(seed)
        x, m = tf(img, mask)
        cy_i, cx_i = centroid(x[0].numpy())
        cy_m, cx_m = centroid(np.asarray(m, np.float32))
        worst = max(worst, float(np.hypot(cy_i - cy_m, cx_i - cx_m)))
    # Zwei Bildpunkte auf 224 sind knapp ein Prozent der Kantenlaenge und
    # decken die Rundung zweier Neuabtastungen ab. Die kaputte Fassung landet
    # weit darueber, siehe den Gegentest.
    check("Bild und Maske bewegen sich gemeinsam", worst < 2.0,
          f"groesster Abstand ueber 16 Ziehungen {worst:.2f} Bildpunkte")


def test_kastenfalle_bei_phase6_staerke() -> None:
    """Dieselbe Pruefung bei der Augmentierung von Phase 6.

    Der stille Fehler ist derselbe wie oben, aber er wiegt mehr: Phase 6 zieht
    die Verschiebung von 3 auf 8 Prozent und die Skalierung von 0,93 bis 1,07
    auf 0,75 bis 1,00. Wer Bild und Maske getrennt zieht, liegt dort um ein
    Vielfaches weiter daneben, und der Verlust faellt trotzdem. Die Roadmap
    verlangt deshalb ausdruecklich, den Rauchtest hier zu wiederholen.

    Zusaetzlich wird nachgerechnet, dass der Aufruf OHNE Argumente bitgleich
    das liefert, was er vor dem Einbau der Schalter lieferte. Sonst waeren alle
    Laeufe bis Phase 5 nicht mehr mit den kommenden vergleichbar.
    """
    print("\ntest_kastenfalle_bei_phase6_staerke")
    img, mask = synthetic_pair()

    torch.manual_seed(5)
    a = TrainTransform(SIZE)(img)
    torch.manual_seed(5)
    b = TrainTransform(SIZE, 0.03, (0.93, 1.07), 7.0)(img)
    check("die Vorgaben sind die alten Werte, bitgleich",
          bool(torch.equal(a, b)))

    tf = TrainTransform(SIZE, translate=0.08, scale=(0.75, 1.0))
    worst = 0.0
    for seed in range(16):
        torch.manual_seed(seed)
        x, m = tf(img, mask)
        cy_i, cx_i = centroid(x[0].numpy())
        cy_m, cx_m = centroid(np.asarray(m, np.float32))
        worst = max(worst, float(np.hypot(cy_i - cy_m, cx_i - cx_m)))
    check("Bild und Maske bewegen sich auch bei Phase-6-Staerke gemeinsam",
          worst < 2.0, f"groesster Abstand ueber 16 Ziehungen {worst:.2f} "
                       f"Bildpunkte")

    # Der Gegentest, und hier ist er wichtiger als oben: die staerkere
    # Augmentierung muss den Fehler auch deutlicher zeigen, sonst haette der
    # Test bei Phase-6-Staerke weniger Kraft als bei Phase-5-Staerke.
    ds = {}
    for name, (tr, sc) in (("phase5", (0.03, (0.93, 1.07))),
                           ("phase6", (0.08, (0.75, 1.0)))):
        aff = T.RandomAffine(degrees=7, translate=(tr, tr), scale=sc)
        d = []
        for seed in range(24):
            torch.manual_seed(seed)
            i2 = aff(TF.resize(img, [SIZE, SIZE]))
            m2 = aff(mask)
            cy_i, cx_i = centroid(np.asarray(i2, np.float32))
            cy_m, cx_m = centroid(np.asarray(m2, np.float32))
            d.append(float(np.hypot(cy_i - cy_m, cx_i - cx_m)))
        ds[name] = float(np.mean(d))
    check("zwei getrennte Ziehungen fallen bei Phase-6-Staerke staerker auf",
          ds["phase6"] > ds["phase5"],
          f"{ds['phase6']:.2f} gegen {ds['phase5']:.2f} Bildpunkte")


def test_gegentest_zwei_ziehungen() -> None:
    """Die kaputte Fassung MUSS auffallen, sonst prueft der Test oben nichts.

    Zwei getrennte `T.RandomAffine`-Aufrufe ziehen zweimal. Genau das ist der
    Fehler, gegen den `TrainTransform` gebaut ist. Gemittelt und nicht je
    Ziehung geprueft: eine einzelne kann zufaellig fast dieselbe sein, dann
    waere der Test launisch und wuerde irgendwann abgeschaltet statt geglaubt.
    """
    print("\ntest_gegentest_zwei_ziehungen")
    img, mask = synthetic_pair()
    aff = T.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.93, 1.07))
    ds = []
    for seed in range(24):
        torch.manual_seed(seed)
        i2 = aff(TF.resize(img, [SIZE, SIZE]))       # erste Ziehung
        m2 = aff(mask)                               # zweite Ziehung, der Fehler
        cy_i, cx_i = centroid(np.asarray(i2, np.float32))
        cy_m, cx_m = centroid(np.asarray(m2, np.float32))
        ds.append(float(np.hypot(cy_i - cy_m, cx_i - cx_m)))
    check("zwei getrennte Ziehungen fallen durch", float(np.mean(ds)) > 2.0,
          f"mittlerer Abstand {np.mean(ds):.2f} Bildpunkte")


def test_zufallsstrom_unabhaengig_von_der_maske() -> None:
    """Mit und ohne Maske muss das BILD identisch herauskommen.

    Das ist die Bedingung dafuer, dass der Arm mit Kopf und der Arm ohne Kopf
    gepaart verglichen werden duerfen: sie sehen dieselben augmentierten
    Bilder, weil die Maske keine einzige Zufallszahl verbraucht.
    """
    print("\ntest_zufallsstrom_unabhaengig_von_der_maske")
    img, mask = synthetic_pair()
    tf = TrainTransform(SIZE)
    torch.manual_seed(11)
    x_alone = tf(img)
    torch.manual_seed(11)
    x_with, _ = tf(img, mask)
    check("die Maske verbraucht keinen Zufall", bool(torch.equal(x_alone, x_with)))


def test_weiches_ziel() -> None:
    print("\ntest_weiches_ziel")
    mask = box_mask_pil(BOX, SIZE)
    field = tile_coverage(mask, HEAD_GRID)
    check("Form ist 1 x grid x grid", tuple(field.shape) == (1, HEAD_GRID, HEAD_GRID),
          str(tuple(field.shape)))
    check("Werte liegen in [0, 1]",
          float(field.min()) >= 0.0 and float(field.max()) <= 1.0)
    a = float(np.asarray(mask, np.float32).mean() / 255.0)
    check("Mittelwert ist der Flaechenanteil des Kastens",
          abs(float(field.mean()) - a) < 1e-5,
          f"Feld {float(field.mean()):.6f} gegen Maske {a:.6f}")
    # Waere das Ziel unbemerkt hart geworden, gaebe es keine Zwischenwerte.
    f = field.numpy().ravel()
    n_soft = int(((f > 0.01) & (f < 0.99)).sum())
    check("es gibt echte Zwischenwerte, das Ziel ist weich", n_soft >= 4,
          f"{n_soft} Kacheln strikt zwischen 0 und 1")
    leer = tile_coverage(box_mask_pil(None, SIZE), HEAD_GRID)
    check("ohne Kasten ist das Feld leer", float(leer.abs().max()) == 0.0)


def test_pos_weight_je_kachel() -> None:
    """`empty` verduennt die positiven Kacheln und braucht ein groesseres Gewicht.

    Genau hier waere der stille Fehler: den fuer `exclude` gerechneten Wert im
    `empty`-Arm zu verwenden. Der Kopf lernte dann, ueberall Null zu sagen, was
    einen hervorragenden Verlust ergibt und wertlos ist.
    """
    print("\ntest_pos_weight_je_kachel")
    boxes = {"a": BOX, "b": BOX}
    ids = ["a", "b", "c", "d", "e", "f", "g", "h"]      # 2 von 8 mit Kasten
    w_ex, cov_ex = tile_pos_weight(boxes, ids, HEAD_GRID, "exclude", SIZE)
    w_em, cov_em = tile_pos_weight(boxes, ids, HEAD_GRID, "empty", SIZE)
    check("empty hat die kleinere Bedeckung", cov_em < cov_ex,
          f"{cov_em:.4f} gegen {cov_ex:.4f}")
    check("empty hat das groessere Gewicht", w_em > w_ex,
          f"{w_em:.2f} gegen {w_ex:.2f}")
    # Eine Richtungspruefung allein wuerde einen Faktor-zwei-Fehler nicht sehen.
    check("die Bedeckung faellt um genau den Faktor 4 (2 von 8 Bildern)",
          abs(cov_ex / cov_em - 4.0) < 1e-6, f"Faktor {cov_ex / cov_em:.6f}")
    check("das Gewicht ist (1 - Bedeckung) / Bedeckung",
          abs(w_ex - (1 - cov_ex) / cov_ex) < 1e-9)


def test_kopfraster_fest() -> None:
    """Der Kern der Phase: das Lineal aendert sich nicht mehr mit dem Gemessenen."""
    print("\ntest_kopfraster_fest")
    net = TwoHeadNet(HEAD_GRID, pretrained=False).eval()
    for size in (224, 320, 448):
        with torch.no_grad():
            logit, field = net(torch.zeros(2, 3, size, size))
        check(f"bei {size} Bildpunkten bleibt das Feld {HEAD_GRID} x {HEAD_GRID}",
              tuple(field.shape) == (2, 1, HEAD_GRID, HEAD_GRID),
              str(tuple(field.shape)))
        check(f"bei {size} Bildpunkten bleibt der Klassenausgang eine Zahl",
              tuple(logit.shape) == (2, 1))


def test_klassifikationsweg_unveraendert() -> None:
    """Der zweikoepfige Rumpf muss bitgleich dasselbe rechnen wie der einkoepfige.

    Sonst unterscheiden sich die beiden Arme in ZWEI Dingen, dem Kopf und dem
    Klassifikationsweg, und der gepaarte Vergleich misst nichts Bestimmtes
    mehr. Geprueft wird nicht "aehnlich", sondern identisch.
    """
    print("\ntest_klassifikationsweg_unveraendert")
    import torch.nn as nn
    from torchvision.models import resnet18

    net = TwoHeadNet(HEAD_GRID, pretrained=False).eval()
    plain = resnet18(weights=None)
    plain.fc = nn.Linear(plain.fc.in_features, 1)
    plain.load_state_dict(net.trunk.state_dict())
    plain.eval()

    torch.manual_seed(0)
    x = torch.randn(3, 3, SIZE, SIZE)
    with torch.no_grad():
        a, b = net(x)[0], plain(x)
    check("zweikoepfig und einkoepfig liefern denselben Logit",
          bool(torch.equal(a, b)),
          f"groesste Abweichung {float((a - b).abs().max()):.3e}")


def test_kopf_startet_neutral() -> None:
    """Zu Beginn steht das Feld ueberall auf Wahrscheinlichkeit 0.5.

    Der neutrale Start ist die Voraussetzung dafuer, dass das auf dem ersten
    Stapel gemessene lambda etwas bedeutet. Startete der Kopf voreingenommen,
    waere sein Anfangsverlust eine Eigenschaft der Initialisierung und nicht
    der Aufgabe.
    """
    print("\ntest_kopf_startet_neutral")
    net = TwoHeadNet(HEAD_GRID, pretrained=False).eval()
    with torch.no_grad():
        field = net(torch.zeros(1, 3, SIZE, SIZE))[1]
    check("das Feld startet bei Logit 0", float(field.abs().max()) < 0.05,
          f"groesster Betrag {float(field.abs().max()):.4f}")


def test_lambda_braucht_einen_kasten_im_stapel() -> None:
    """Ein Stapel ohne annotiertes Bild darf lambda nicht setzen.

    Der stille Fehler dahinter: bei `exclude` traegt ein Bild ohne Kasten
    nichts zum Kopfverlust bei. Enthaelt der Stapel gar keinen Kasten, ist der
    Verlust exakt null, und aus dem Verhaeltnis der beiden Verluste wird das,
    was der Abfang bei 1e-8 daraus macht. Nichts bricht, nichts warnt, der
    Rumpf trainiert acht Epochen lang auf die falsche Mischung.
    """
    print("\ntest_lambda_braucht_einen_kasten_im_stapel")
    import torch.nn as nn

    B = 4
    crit_loc = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([7.5]), reduction="none")
    torch.manual_seed(3)
    field = torch.randn(B, 1, HEAD_GRID, HEAD_GRID)
    target = torch.rand(B, 1, HEAD_GRID, HEAD_GRID)

    leer = torch.zeros(B)                       # kein Bild traegt einen Kasten
    voll = torch.tensor([0.0, 1.0, 0.0, 1.0])   # zwei tragen einen

    check("ein Stapel ohne Kasten darf lambda nicht setzen",
          not batch_can_set_lambda(leer))
    check("ein Stapel mit Kasten darf es", batch_can_set_lambda(voll))

    # Die Bedingung ist nur dann gratis, wenn der uebersprungene Stapel
    # ohnehin nichts beitraegt. Genau das wird hier nachgerechnet und nicht
    # angenommen.
    l_loc_leer = loc_loss(field, target, leer, crit_loc)
    check("ohne Kasten ist der Kopfverlust exakt null",
          float(l_loc_leer) == 0.0, f"{float(l_loc_leer):.3e}")
    l_cls = torch.tensor(0.7505241)
    mit_null = l_cls + 0.0 * l_loc_leer
    mit_kaputt = l_cls + 7.5052408e7 * l_loc_leer
    check("der Verlust dieses Stapels haengt nicht an lambda",
          bool(torch.equal(mit_null, mit_kaputt)),
          "warten kostet diesen Stapel nichts")

    # Der Gegentest. Ohne ihn wuerde der Test oben auch die kaputte Fassung
    # durchwinken, denn die setzt lambda ebenfalls, nur eben aus null.
    alt = float((l_cls / l_loc_leer.clamp(min=1e-8)).cpu())
    check("die alte Regel faellt durch den Bereichswaechter",
          not (LAMBDA_MIN <= alt <= LAMBDA_MAX), f"lambda waere {alt:.4g}")

    # Und die vierzehn heilen Laeufe muessen drinbleiben, sonst waere der
    # Waechter zu eng und wuerde gesunde Laeufe abbrechen.
    gemessen = [0.9688, 1.2687, 0.9473, 1.1795,
                0.9562, 1.0712, 1.2569, 0.7082, 0.8443]
    check("die gemessenen lambda der Phase 5 liegen im Bereich",
          all(LAMBDA_MIN <= v <= LAMBDA_MAX for v in gemessen),
          f"{min(gemessen):.4f} bis {max(gemessen):.4f} gegen "
          f"[{LAMBDA_MIN:g}, {LAMBDA_MAX:g}]")


if __name__ == "__main__":
    test_kastenfalle()
    test_kastenfalle_bei_phase6_staerke()
    test_gegentest_zwei_ziehungen()
    test_zufallsstrom_unabhaengig_von_der_maske()
    test_weiches_ziel()
    test_pos_weight_je_kachel()
    test_kopfraster_fest()
    test_klassifikationsweg_unveraendert()
    test_kopf_startet_neutral()
    test_lambda_braucht_einen_kasten_im_stapel()
    print("\n" + ("ALLE TESTS BESTANDEN" if not FAILED
                  else f"{len(FAILED)} FEHLGESCHLAGEN: {FAILED}"))
    sys.exit(1 if FAILED else 0)
