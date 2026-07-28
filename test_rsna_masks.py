"""
Prueft die Nicht-Torch-Logik von rsna_make_masks.py und rsna_cam_lung_check.py.

Warum ausgerechnet diese Teile: die Diagnose "Maximum ausserhalb der Lunge"
besteht aus drei Rasterumrechnungen (Box 1024 -> 224, Maske 256 -> 224, Heatmap
224) und einer Bedingungslogik. Ein Faktorfehler oder eine vertauschte
Achse faellt in der fertigen Zahl NICHT auf -- "38 % der Fehlschlaege liegen
ausserhalb der Lunge" sieht plausibel aus, egal ob die Maske an der richtigen
Stelle liegt. Also werden Geometrie und Zuordnung hier gegen von Hand gebaute
Faelle geprueft, bei denen die Antwort vorher feststeht.

Beide Module importieren Torch nur innerhalb der Rechenfunktionen. Dieser Test
kommt deshalb ohne Torch aus -- absichtlich, damit er auch dann laeuft, wenn
gerade ein Training die GPU belegt.

  python test_rsna_masks.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import rsna_cam_lung_check as clc
import rsna_make_masks as mm

FAILED: list[str] = []


def check(name: str, cond: bool, info: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FAILED.append(name)


def close(a, b, tol=1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------

def test_box_geometry() -> None:
    print("\nBoxgeometrie 1024 -> 224")

    # Ganzes Bild: muss die volle Flaeche ergeben.
    m = clc.box_mask([(0, 0, 1024, 1024)], 224, 1024)
    check("volle Box deckt alles", m.all() and close(m.mean(), 1.0))

    # Oberes linkes Viertel im Originalraster -> oberes linkes Viertel bei 224.
    m = clc.box_mask([(0, 0, 512, 512)], 224, 1024)
    check("Viertel-Box -> Flaeche 0,25", close(m.mean(), 0.25),
          f"gemessen {m.mean():.4f}")
    check("Viertel-Box liegt oben links",
          m[0, 0] and m[111, 111] and not m[112, 112] and not m[223, 223])

    # Achsen nicht vertauscht: schmal in x, hoch in y.
    m = clc.box_mask([(0, 0, 1024, 512)], 224, 1024)   # volle Breite, halbe Hoehe
    check("x/y nicht vertauscht", m[0, :].all() and not m[223, :].any(),
          "volle Breite oben, unten leer")

    # Negative Koordinaten werden geklemmt statt umzulaufen (Python-Slicing
    # mit negativem Start waere sonst ein stiller Fehler am Bildrand).
    m = clc.box_mask([(-100, -100, 300, 300)], 224, 1024)
    check("negativer Ursprung wird geklemmt", m[0, 0] and m.sum() > 0)

    # Zwei Boxen vereinigen sich, ueberlappende Flaeche wird nicht doppelt gezaehlt.
    m2 = clc.box_mask([(0, 0, 512, 512), (0, 0, 512, 512)], 224, 1024)
    check("doppelte Box zaehlt einfach", close(m2.mean(), 0.25))


def test_analyse_one() -> None:
    print("\nanalyse_one: Zuordnung Maximum -> Box / Lunge")
    size = 20
    box = np.zeros((size, size), bool); box[2:6, 2:6] = True
    lung = np.zeros((size, size), bool); lung[0:10, 0:10] = True

    # Fall A: Maximum in Box und in Lunge.
    heat = np.zeros((size, size)); heat[3, 3] = 1.0
    r = clc.analyse_one(heat, box, lung)
    check("A Maximum in Box", r["peak_in_box"])
    check("A Maximum in Lunge", r["peak_in_lung"])
    check("A Masse in Box = 1", close(r["mass_in_box"], 1.0))

    # Fall B: Maximum ausserhalb der Lunge (Bildrand), zweitbester Punkt in der
    # Box. Genau der Fall, fuer den ein Crop etwas braechte.
    heat = np.zeros((size, size)); heat[19, 19] = 1.0; heat[3, 3] = 0.9
    r = clc.analyse_one(heat, box, lung)
    check("B Maximum nicht in Box", not r["peak_in_box"])
    check("B Maximum nicht in Lunge", not r["peak_in_lung"])
    check("B lungenbeschraenkt trifft die Box", r["peak_in_box_lungrestricted"],
          "-> Spielraum fuer Crop")

    # Fall C: Maximum in der Lunge, aber neben der Box. Ein Crop aendert nichts.
    heat = np.zeros((size, size)); heat[8, 8] = 1.0; heat[3, 3] = 0.5
    r = clc.analyse_one(heat, box, lung)
    check("C Maximum nicht in Box", not r["peak_in_box"])
    check("C Maximum in der Lunge", r["peak_in_lung"])
    check("C lungenbeschraenkt trifft weiterhin nicht",
          not r["peak_in_box_lungrestricted"], "-> kein Spielraum")

    # Flaechen und box_in_lung: Box liegt voll in der Lunge -> 1,0.
    check("box_in_lung = 1 bei voller Ueberdeckung", close(r["box_in_lung"], 1.0))
    check("lung_area korrekt", close(r["lung_area"], 100 / 400))
    check("box_area korrekt", close(r["box_area"], 16 / 400))

    # Halb ueberdeckte Box -> 0,5. Das ist die Kontrollzahl gegen
    # Untersegmentierung; sie muss stimmen.
    lung_half = np.zeros((size, size), bool); lung_half[0:4, :] = True
    r = clc.analyse_one(heat, box, lung_half)
    check("box_in_lung = 0,5 bei halber Ueberdeckung",
          close(r["box_in_lung"], 0.5), f"gemessen {r['box_in_lung']:.3f}")

    # Leere Heatmap -> None statt Division durch Null.
    check("leere Heatmap ergibt None",
          clc.analyse_one(np.zeros((size, size)), box, lung) is None)

    # Negative CAM-Werte werden geklemmt, nicht als Masse gezaehlt.
    heat = np.zeros((size, size)); heat[3, 3] = 1.0; heat[15, 15] = -5.0
    r = clc.analyse_one(heat, box, lung)
    check("negative Werte werden geklemmt", close(r["mass_in_box"], 1.0))

    # Leere Lungenmaske: keine Einschraenkung, kein Absturz.
    r = clc.analyse_one(heat, box, np.zeros((size, size), bool))
    check("leere Maske faellt auf ungebremstes Maximum zurueck",
          r["peak_in_box_lungrestricted"] == r["peak_in_box"])
    check("leere Maske -> lung_area 0", close(r["lung_area"], 0.0))


def test_summarise() -> None:
    print("\nsummarise: Aufschluesselung der Fehlschlaege")
    def row(hit, in_lung, restr, mass_lung):
        return dict(peak_in_box=hit, peak_in_lung=in_lung,
                    peak_in_box_lungrestricted=restr,
                    mass_in_box=0.5 if hit else 0.1, mass_in_lung=mass_lung,
                    box_area=0.2, lung_area=0.4, box_in_lung=1.0,
                    null_free=0.2, null_restricted=0.5)

    # 2 Treffer, 3 Fehlschlaege -- davon 2 ausserhalb der Lunge.
    df = pd.DataFrame([
        row(True,  True,  True,  0.8),
        row(True,  True,  True,  0.8),
        row(False, False, True,  0.3),
        row(False, False, False, 0.3),
        row(False, True,  False, 0.9),
    ])
    s = clc.summarise(df)
    check("n", s["n"] == 5)
    check("Trefferquote 0,4", close(s["peak_in_box"], 0.4))
    check("Fehlschlaege gezaehlt", s["n_miss"] == 3)
    check("Fehlschlaege aussen = 2/3", close(s["miss_outside_lung"], 2 / 3),
          f"gemessen {s['miss_outside_lung']:.4f}")
    check("Vorsprung Lunge = 0,6 - 0,4", close(s["peak_in_lung_lift"], 0.2))
    check("Spielraum = 0,6 - 0,4", close(s["crop_headroom"], 0.2),
          "lungenbeschraenkt 3/5 gegen 2/5")

    # Ohne Fehlschlaege darf nichts abstuerzen, sondern NaN kommen.
    s2 = clc.summarise(df[df.peak_in_box].reset_index(drop=True))
    check("keine Fehlschlaege -> NaN statt Absturz", np.isnan(s2["miss_outside_lung"]))


def test_cv_mean() -> None:
    print("\ncv_mean: CV-Mittel ueber Folds")
    rows = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    m, s = clc.cv_mean(rows, "x")
    check("Mittel", close(m, 2.0))
    check("SD mit ddof=1", close(s, 1.0))
    m, s = clc.cv_mean([{"x": float("nan")}, {"x": 4.0}], "x")
    check("NaN wird uebergangen", close(m, 4.0))
    m, _ = clc.cv_mean([{"x": float("nan")}], "x")
    check("nur NaN -> NaN", np.isnan(m))


def test_ids_from_csvs() -> None:
    print("\nrsna_make_masks: ID-Auswahl")
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        pd.DataFrame({"patientId": ["b", "a", "a"]}).to_csv(t / "cam_f0_s0.csv",
                                                           index=False)
        pd.DataFrame({"patientId": ["c", "a"]}).to_csv(t / "cam_f1_s0.csv",
                                                       index=False)
        ids = mm.ids_from_csvs([str(t / "cam_f*_s0.csv")])
        check("Glob erfasst beide Folds, dedupliziert, sortiert",
              ids == ["a", "b", "c"], str(ids))

        try:
            mm.ids_from_csvs([str(t / "gibtsnicht_*.csv")])
            check("leeres Glob wirft", False)
        except FileNotFoundError:
            check("leeres Glob wirft FileNotFoundError", True)

        pd.DataFrame({"foo": [1]}).to_csv(t / "bad.csv", index=False)
        try:
            mm.ids_from_csvs([str(t / "bad.csv")])
            check("fehlende Spalte wirft", False)
        except ValueError:
            check("fehlende Spalte wirft ValueError", True)


def test_pending_jobs() -> None:
    print("\nrsna_make_masks: was ist noch zu rechnen")
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "src", Path(tmp) / "dst"
        src.mkdir(); dst.mkdir()
        for pid in ("a", "b", "c"):
            (src / f"{pid}.png").write_bytes(b"x")
        (dst / "a.png").write_bytes(b"x")            # schon vorhanden

        jobs, skipped, missing = mm.pending_jobs(["a", "b", "c", "weg"], src, dst,
                                                 overwrite=False)
        check("vorhandene Maske wird uebersprungen", skipped == 1)
        check("fehlendes Quellbild wird gezaehlt", missing == 1)
        check("zwei Bilder zu rechnen", len(jobs) == 2,
              str([j[0].stem for j in jobs]))

        jobs, skipped, _ = mm.pending_jobs(["a", "b", "c"], src, dst, overwrite=True)
        check("--overwrite rechnet alles neu", len(jobs) == 3 and skipped == 0)


def test_area_report() -> None:
    print("\nrsna_make_masks: Flaechenstatistik")
    a = np.array([0.01, 0.30, 0.35, 0.40, 0.90])
    r = mm.area_report(a)
    check("n", r["n"] == 5)
    check("Median", close(r["median"], 0.35))
    check("leere Maske erkannt (<0,05)", r["n_empty"] == 1)
    check("riesige Maske erkannt (>0,60)", r["n_huge"] == 1)
    check("Grenzwerte sind strikt: 0,05 zaehlt nicht als leer",
          mm.area_report(np.array([0.05, 0.05]))["n_empty"] == 0)
    check("leeres Array -> leeres dict", mm.area_report(np.array([])) == {})


def test_packing() -> None:
    print("\nRoh-Cache: Bit-Packing")
    rng = np.random.default_rng(0)
    m = rng.random((7, 256, 256)) > 0.5
    back = mm.unpack_masks(mm.pack_masks(m))
    check("Rundlauf verlustfrei", bool((back == m).all()))
    check("Faktor 8 kleiner", mm.pack_masks(m).nbytes * 8 == m.nbytes,
          f"{mm.pack_masks(m).nbytes} vs {m.nbytes} Byte")
    # Ein Bit-Offset waere der klassische stille Fehler: die Maske sieht noch
    # plausibel aus, ist aber um eine Zeile verschoben.
    one = np.zeros((1, 256, 256), bool); one[0, 3, 5] = True
    b = mm.unpack_masks(mm.pack_masks(one))[0]
    check("Position bleibt exakt erhalten",
          b[3, 5] and b.sum() == 1, f"Summe {b.sum()}")


def test_refine_variant() -> None:
    print("\nVerfeinerungs-Varianten")
    raw = np.zeros((256, 256), bool)
    raw[60:200, 40:100] = True          # linke Lunge
    raw[60:200, 156:216] = True         # rechte Lunge

    a_none = mm.refine_variant(raw, "none", 0).mean()
    a_def = mm.refine_variant(raw, "default", 0).mean()
    a_hull = mm.refine_variant(raw, "hull", 0).mean()
    a_dil = mm.refine_variant(raw, "default", 6).mean()
    check("Dilatation vergroessert die Flaeche", a_dil > a_def,
          f"{a_def:.4f} -> {a_dil:.4f}")
    check("Huelle ist nie kleiner als default", a_hull >= a_def - 1e-9,
          f"{a_def:.4f} vs {a_hull:.4f}")
    check("'none' laesst die Rohmaske unveraendert", close(a_none, raw.mean()))
    check("Dilatation ist monoton",
          mm.refine_variant(raw, "default", 2).mean()
          <= mm.refine_variant(raw, "default", 8).mean())

    # Der Konsolidierungs-Fall, in zwei Ausfuehrungen -- und die beiden
    # verhalten sich verschieden. Das ist keine Spitzfindigkeit, sondern
    # bestimmt, ob 'hull' als Gegenmittel ueberhaupt taugt:
    #
    #   a) EINGESCHLOSSENES Loch mitten in der Lunge. Das fuellt schon
    #      `_clean` per binary_fill_holes -- 'hull' aendert nichts.
    #   b) RANDOFFENE Einbuchtung (die Konsolidierung reicht bis zur
    #      Pleura/Zwerchfell). fill_holes kann das nicht, weil die Aussparung
    #      mit dem Hintergrund verbunden ist. Nur die konvexe Huelle holt sie
    #      zurueck. Genau diese Form hat eine echte Unterlappen-Pneumonie.
    enclosed = raw.copy(); enclosed[100:150, 50:90] = False
    check("eingeschlossenes Loch: schon 'default' fuellt es",
          close(mm.refine_variant(enclosed, "default", 0).mean(), a_def),
          "-> 'hull' ist dafuer nicht noetig")

    notched = raw.copy(); notched[100:150, 40:90] = False      # reicht an den Rand
    n_def = mm.refine_variant(notched, "default", 0).mean()
    n_hull = mm.refine_variant(notched, "hull", 0).mean()
    check("randoffene Einbuchtung: 'default' bekommt sie nicht zurueck",
          n_def < a_def - 1e-6, f"{n_def:.4f} < {a_def:.4f}")
    check("randoffene Einbuchtung: 'hull' holt Flaeche zurueck", n_hull > n_def,
          f"{n_def:.4f} -> {n_hull:.4f}")

    try:
        mm.refine_variant(raw, "quatsch", 0)
        check("unbekannter Modus wirft", False)
    except ValueError:
        check("unbekannter Modus wirft ValueError", True)

    out = mm.to_out(mm.refine_variant(raw, "default", 0))
    check("to_out liefert 224x224", out.shape == (224, 224), str(out.shape))
    check("to_out bleibt binaer", set(np.unique(out)) <= {0, 255},
          str(np.unique(out)))


def test_null_baselines() -> None:
    print("\nNullhypothesen (der Fehler der ersten Fassung)")
    size = 20
    box = np.zeros((size, size), bool); box[2:6, 2:6] = True      # Flaeche 0,04
    lung = np.zeros((size, size), bool); lung[0:10, 0:10] = True  # Flaeche 0,25
    heat = np.zeros((size, size)); heat[3, 3] = 1.0
    r = clc.analyse_one(heat, box, lung)

    check("null_free = Boxflaeche", close(r["null_free"], 0.04))
    # Box liegt ganz in der Lunge -> Zufallstreffer in der Lunge = 0,04/0,25
    check("null_restricted = (Box UND Lunge)/Lunge",
          close(r["null_restricted"], 0.04 / 0.25),
          f"gemessen {r['null_restricted']:.4f}")
    check("Peak-Koordinaten werden gespeichert",
          r["peak_y"] == 3 and r["peak_x"] == 3)

    # Ein Fold, in dem der Aussen-Anteil GENAU dem Zufall entspricht: der
    # Vorsprung muss 0 sein, nicht der Rohwert gross aussehen.
    n = 100
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n):
        rows.append(dict(peak_in_box=False, peak_in_lung=bool(rng.random() < 0.25),
                         peak_in_box_lungrestricted=False, mass_in_box=0.1,
                         mass_in_lung=0.25, box_area=0.04, lung_area=0.25,
                         box_in_lung=1.0, null_free=0.04, null_restricted=0.16))
    s = clc.summarise(pd.DataFrame(rows))
    check("Aussen-Rohwert ist gross (~0,75)", s["miss_outside_lung"] > 0.6,
          f"{s['miss_outside_lung']:.3f}")
    check("Vorsprung gegen die Null ist ~0", abs(s["miss_outside_lift"]) < 0.12,
          f"{s['miss_outside_lift']:+.3f}  <- genau die Zahl, die gefehlt hat")
    check("headroom_null wird berechnet", close(s["headroom_null"], 0.12),
          f"{s['headroom_null']:.3f}")


def test_paired_t() -> None:
    print("\ngepaartes t ueber Folds")
    rows = [{"d": 0.08}, {"d": 0.07}, {"d": 0.09}, {"d": 0.08}, {"d": 0.08}]
    check("konsistente Differenz -> grosses t", clc.paired_t(rows, "d") > 2.78,
          f"t={clc.paired_t(rows, 'd'):.2f}")
    rows = [{"d": -0.12}, {"d": -0.06}, {"d": 0.05}, {"d": 0.12}, {"d": 0.08}]
    check("streuende Differenz -> kleines t", abs(clc.paired_t(rows, "d")) < 2.78,
          f"t={clc.paired_t(rows, 'd'):.2f}   (der reale Fall)")
    check("ein Wert -> NaN", np.isnan(clc.paired_t([{"d": 1.0}], "d")))


def test_parse_variant() -> None:
    print("\nSweep: Variantenparser")
    import rsna_mask_sweep as sw
    check("'hull:4'", sw.parse_variant("hull:4") == ("hull", 4))
    check("ohne Doppelpunkt -> 0 Pixel", sw.parse_variant("default") == ("default", 0))
    check("'none:0'", sw.parse_variant("none:0") == ("none", 0))


def test_balanced_sample() -> None:
    print("\nrsna_make_masks: ausgewogene Stichprobe")
    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "splits.json"
        labels = {f"p{i}": (1 if i < 20 else 0) for i in range(100)}
        holdout = ["p0", "p1", "p90", "p91"]
        sp.write_text(json.dumps({"labels": labels, "holdout": holdout}))

        got = mm.balanced_sample(sp, 5, seed=0)
        check("2x5 Bilder", len(got) == 10, str(len(got)))
        pos = sum(labels[g] for g in got)
        check("ausgewogen 5/5", pos == 5, f"{pos} positiv")
        check("Holdout ausgeschlossen", not (set(got) & set(holdout)),
              "-- sonst waere die reservierte Menge still angefasst")
        check("deterministisch bei gleichem Seed",
              mm.balanced_sample(sp, 5, seed=0) == got)
        check("anderer Seed -> andere Auswahl",
              mm.balanced_sample(sp, 5, seed=1) != got)

        # Mehr angefordert als vorhanden: nehmen, was da ist, statt zu werfen.
        many = mm.balanced_sample(sp, 1000, seed=0)
        check("Ueberanforderung wird gekappt", len(many) == 96,
              f"{len(many)} von 100 minus 4 Holdout")


def test_load_boxes_matches_train() -> None:
    """Der Sweep hat eine eigene Kopie von load_boxes, damit er ohne Torch
    laeuft. Diese Pruefung verhindert, dass die beiden auseinanderdriften."""
    print("\nSweep: load_boxes-Kopie")
    import rsna_mask_sweep as sw
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        pd.DataFrame({
            "patientId": ["a", "a", "b", "c", "d"],
            "x": [10, 50, 20, np.nan, 5],
            "y": [10, 50, 20, 30, 5],
            "width": [100, 100, 200, 40, 10],
            "height": [100, 100, 200, 40, 10],
            "Target": [1, 1, 1, 1, 0],
        }).to_csv(t / "stage_2_train_labels.csv", index=False)
        got = sw.load_boxes(t)
        check("nur Target==1 mit gueltigen Koordinaten",
              set(got) == {"a", "b"}, str(sorted(got)))
        check("zwei Boxen fuer 'a'", len(got["a"]) == 2)
        check("Zeile mit NaN-Koordinate fliegt raus", "c" not in got)
        check("Negativer (Target=0) fliegt raus", "d" not in got)
        check("Koordinaten als float", isinstance(got["b"][0][0], float))

        try:
            from rsna_train import load_boxes as orig
        except Exception as e:                       # kein Torch -> nicht pruefbar
            print(f"  --    Vergleich mit rsna_train uebersprungen ({type(e).__name__})")
            return
        check("identisch mit rsna_train.load_boxes", orig(t) == got)
        from rsna_train import BOX_SPACE as ORIG_SPACE
        check("BOX_SPACE identisch", sw.BOX_SPACE == ORIG_SPACE)


def test_grouped_bootstrap() -> None:
    """Der gruppierte Bootstrap muss ein WEITERES Intervall liefern als der
    bildweise -- sonst greift die Gruppierung nicht, und die externe Zahl
    saehe sicherer aus, als sie ist."""
    print("\nExtern: gruppierter Bootstrap")
    import rsna_external_kermany as ext

    rng = np.random.default_rng(0)
    n_groups, per_group = 40, 10
    groups, y, p = [], [], []
    for g in range(n_groups):
        cls = g % 2
        # Innerhalb einer Gruppe fast identische Scores: so sehen mehrere
        # Aufnahmen desselben Kindes aus.
        base = rng.normal(0.6 if cls else 0.4, 0.25)
        for _ in range(per_group):
            groups.append(f"g{g}"); y.append(cls)
            p.append(base + rng.normal(0, 0.01))
    y, p, groups = np.array(y), np.array(p), np.array(groups)

    a_g, lo_g, hi_g = ext.grouped_bootstrap_auc(y, p, groups, B=200, seed=0)
    a_i, lo_i, hi_i = ext.grouped_bootstrap_auc(
        y, p, np.arange(len(y)).astype(str), B=200, seed=0)   # jede Zeile eigene Gruppe
    check("Punktschaetzer identisch", close(a_g, a_i, 1e-9))
    check("gruppiertes Intervall ist WEITER", (hi_g - lo_g) > (hi_i - lo_i),
          f"gruppiert {hi_g - lo_g:.3f} vs bildweise {hi_i - lo_i:.3f}")


def test_pad_to_square() -> None:
    print("\nExtern: PadToSquare")
    from PIL import Image as PILImage

    import rsna_external_kermany as ext

    a = np.zeros((60, 100), np.uint8); a[:] = 200
    img = PILImage.fromarray(a, mode="L")
    out = ext.PadToSquare()(img)
    check("wird quadratisch", out.size == (100, 100), str(out.size))
    o = np.asarray(out)
    check("Bildinhalt zentriert erhalten", bool((o[20:80, :] == 200).all()))
    check("Fuellung ist der Median, nicht Schwarz", int(o[0, 0]) == 200,
          f"Ecke {o[0, 0]} -- Schwarz waere selbst ein Merkmal")
    sq = PILImage.fromarray(np.zeros((50, 50), np.uint8), mode="L")
    check("quadratisches Bild bleibt unveraendert",
          ext.PadToSquare()(sq).size == (50, 50))


def test_stratified_by_score() -> None:
    print("\nExtern: Schichtung nach dem Leak-Score")
    import rsna_external_kermany as ext

    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n)

    # WICHTIG fuer die Testkonstruktion: das Rauschen muss GROSS gegen den
    # Klassenabstand sein. Bei kleinem Rauschen zerfaellt der Score in zwei
    # getrennte Gipfel, die Quintile enthalten dann je nur eine Klasse, und
    # gemessen wird nur noch das Randquintil -- die Schichtung liefe leer.
    # Genau daran ist die erste Fassung dieses Tests gescheitert.
    leak = 1.0 * y + rng.normal(0, 1.0, n)

    raw = roc_auc_score(y, leak)
    a, per = ext.stratified_by_score(y, leak, leak)
    check("Bericht enthaelt n_pos, n_neg und diskordante Paare je Quintil",
          len(per[0]) == 6, str(per[0]))
    check("reiner Leak-Score bricht geschichtet ein", a < raw - 0.15,
          f"roh {raw:.3f} -> geschichtet {a:.3f}")
    check("und landet nahe am Zufall", abs(a - 0.5) < 0.10, f"{a:.3f}")

    # Score mit echtem Zusatzsignal: haelt sich auch innerhalb der Quintile.
    good = leak + 2.0 * y
    raw_g = roc_auc_score(y, good)
    a2, _ = ext.stratified_by_score(y, good, leak)
    check("echtes Zusatzsignal ueberlebt die Schichtung", a2 > 0.70,
          f"roh {raw_g:.3f} -> geschichtet {a2:.3f}")
    check("und liegt klar ueber dem reinen Leak", a2 > a + 0.15,
          f"{a2:.3f} gegen {a:.3f}")

    # Der Fehler aus dem ersten echten Lauf: ein fast reinrassiges Quintil
    # (1182 positiv, 1 negativ) zog mit vollem n-Gewicht ins Mittel. Nach
    # diskordanten Paaren gewichtet, faellt es praktisch heraus.
    n2 = 2000
    y2 = np.concatenate([rng.integers(0, 2, n2), np.ones(n2, int)])
    strat2 = np.concatenate([rng.normal(0, 1, n2), rng.normal(9, 0.1, n2)])
    y2[n2 + 5] = 0                                   # genau EIN Negativer oben
    good = np.where(y2 == 1, 3.0, 0.0) + rng.normal(0, 1, len(y2))
    junk = rng.normal(0, 1, len(y2))                 # oben reines Rauschen
    p2 = np.where(strat2 > 5, junk, good)
    a3, per3 = ext.stratified_by_score(y2, p2, strat2)
    tail = [r for r in per3 if r[3] < 30]            # Quintile mit <30 Negativen
    check("fast reinrassiges Quintil wird erkannt", len(tail) >= 1,
          f"{len(tail)} Quintil(e) mit unter 30 Negativen")
    n_weighted = float(np.average([r[5] for r in per3 if not np.isnan(r[5])],
                                  weights=[r[1] for r in per3
                                           if not np.isnan(r[5])]))
    check("Paar-Gewichtung ignoriert das Rausch-Quintil, n-Gewichtung nicht",
          abs(a3 - 0.5) > abs(n_weighted - 0.5) - 1e-9,
          f"Paare {a3:.3f} vs n-gewichtet {n_weighted:.3f}")


def test_operating_point() -> None:
    print("\nExtern: Arbeitspunkt")
    import rsna_external_kermany as ext

    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([.9, .8, .2, .7, .1, .1, .1, .1])
    op = ext.operating_point(y, p, 0.5)
    check("Sensitivitaet 2/3", close(op["sens"], 2 / 3))
    check("Spezifitaet 4/5", close(op["spec"], 4 / 5))
    check("PPV 2/3", close(op["ppv"], 2 / 3))
    check("Positivrate 3/8", close(op["pos_rate"], 3 / 8))


def test_no_device_map_location() -> None:
    """Kein RSNA-Skript darf einen Checkpoint direkt aufs Geraet laden.

    Das ist eine Quelltextpruefung und keine Verhaltenspruefung -- absichtlich:
    der Fehler tritt NUR mit echter DirectML-Hardware auf, laesst sich in einer
    CPU-Umgebung also nicht ausloesen. Er kostete einen abgebrochenen Lauf und
    meldet sich als

        TypeError: '>=' not supported between instances of 'torch.device' and 'int'

    was nach einem kaputten Checkpoint aussieht. Torch reicht `map_location`
    an `torch_directml.device()` weiter, die dort einen Integer erwartet.
    Richtig ist immer: auf die CPU laden, Modell aufs Geraet, kopieren lassen.
    """
    print("\nDirectML: Checkpoint nie direkt aufs Geraet laden")
    import ast

    # Ueber den AST statt ueber den Rohtext: sonst schlaegt die Pruefung auf
    # den Docstrings an, die den Fehler ERKLAEREN. Gemeint sind ausschliesslich
    # echte Aufrufe mit dem Schluesselwort map_location.
    bad, checked = [], 0
    for f in sorted(Path(".").glob("rsna_*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "map_location":
                    continue
                checked += 1
                ok = isinstance(kw.value, ast.Constant) and kw.value.value == "cpu"
                if not ok:
                    bad.append(f"{f.name}:{kw.value.lineno}")
    check("alle rsna_*.py laden auf 'cpu'", not bad,
          "; ".join(bad) if bad else f"{checked} Aufrufe geprueft")


def test_cache_resume() -> None:
    """Der Fortsetzungspfad des grossen Maskenlaufs.

    Der Fehler, gegen den hier geprueft wird: ein abgebrochener Lauf hat die
    Masken-PNG geschrieben, den Roh-Cache danach aber nicht mehr gesichert.
    Wird beim naechsten Lauf nur auf die PNG geschaut, gilt das Bild als
    erledigt -- und fehlt im Cache fuer immer. Der Zuschnitt merkt das erst
    Stunden spaeter, und dann als fehlendes Bild statt als Fehler.
    """
    print("\nrsna_make_masks: Roh-Cache und Fortsetzung")
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        src, dst = t / "src", t / "dst"
        src.mkdir(); dst.mkdir()
        for pid in ("a", "b", "c"):
            (src / f"{pid}.png").write_bytes(b"x")
            (dst / f"{pid}.png").write_bytes(b"x")   # alle drei "gerechnet"

        cache = t / "raw.npz"
        rng = np.random.default_rng(0)
        masks = rng.random((2, mm.SEG_SIZE, mm.SEG_SIZE)) > 0.5
        mm.save_raw_cache(cache, ["a", "b"], masks)   # aber nur zwei gesichert

        check("cached_ids liest den Cache", mm.cached_ids(cache) == {"a", "b"})
        check("cached_ids ohne Datei -> leer", mm.cached_ids(t / "weg.npz") == set())
        check("cached_ids ohne Pfad -> leer", mm.cached_ids(None) == set())

        jobs, skipped, _ = mm.pending_jobs(["a", "b", "c"], src, dst,
                                           overwrite=False,
                                           cached=mm.cached_ids(cache))
        check("nur das ungesicherte Bild wird nachgerechnet",
              [j[0].stem for j in jobs] == ["c"], str([j[0].stem for j in jobs]))
        check("die gesicherten gelten als erledigt", skipped == 2)

        jobs, skipped, _ = mm.pending_jobs(["a", "b", "c"], src, dst,
                                           overwrite=False, cached=None)
        check("ohne Cache-Fuehrung bleibt das alte Verhalten",
              len(jobs) == 0 and skipped == 3)

        # Nachtragen: 'c' dazu, 'a' und 'b' duerfen nicht verloren gehen.
        mm.save_raw_cache(cache, ["c"],
                          rng.random((1, mm.SEG_SIZE, mm.SEG_SIZE)) > 0.5)
        ids2, packed2 = mm.load_raw_cache(cache)
        check("Nachtragen erhaelt die alten Eintraege", ids2 == ["a", "b", "c"],
              str(ids2))
        check("Inhalt von 'a' unveraendert",
              bool((mm.unpack_masks(packed2[0:1])[0] == masks[0]).all()))
        check("kein .tmp.npz bleibt liegen", list(t.glob("*.tmp.npz")) == [])


def main() -> int:
    for fn in (test_box_geometry, test_analyse_one, test_summarise, test_cv_mean,
               test_ids_from_csvs, test_pending_jobs, test_cache_resume,
               test_area_report,
               test_packing, test_refine_variant, test_null_baselines,
               test_paired_t, test_parse_variant, test_balanced_sample,
               test_load_boxes_matches_train, test_grouped_bootstrap,
               test_pad_to_square, test_stratified_by_score,
               test_operating_point, test_no_device_map_location):
        fn()
    print("\n" + ("-" * 60))
    if FAILED:
        print(f"{len(FAILED)} FEHLGESCHLAGEN: " + ", ".join(FAILED))
        return 1
    print("Alle Pruefungen bestanden.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
