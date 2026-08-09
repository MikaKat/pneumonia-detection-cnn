"""
Phase 5, Schritt 0: wie fein wird das Kopfraster? Gemessen, nicht gewaehlt.

WORUM ES GEHT
-------------
Der zweite Kopf gibt kein einzelnes Ja/Nein aus, sondern ein Feld: fuer jede
Kachel eine Zahl. Wie viele Kacheln das Feld hat, ist eine Entwurfsentscheidung,
und die Roadmap verlangt ausdruecklich, sie zu MESSEN statt zu waehlen
(erklaerungen/00_roadmap_v1.md, Phase 5, "Rasterweite 7 oder 14").

Der Grund, warum das ueberhaupt eine Frage ist: ein Kasten liegt nie sauber auf
Kachelgrenzen. Ein grobes Raster erfindet deshalb Kastenflaeche dazu (Kacheln,
die nur zur Haelfte im Kasten liegen und trotzdem ganz zaehlen) und verliert
welche (Kacheln, die knapp unter der Schwelle bleiben). Das ist ein Fehler, den
das beste denkbare Modell nicht mehr wegbekommt, weil er schon im ZIEL steckt.

WAS GEMESSEN WIRD: DIE DECKE
----------------------------
Statt Zwischengroessen misst dieses Skript direkt die Groesse, in der Phase 5
spaeter berichtet wird: die Punkt-AUC auf dem festen Referenzraster 224
(rsna_lokalisation.point_auc).

Gemessen wird sie an einem gedachten PERFEKTEN Kopf. Das ist ein Kopf, der sein
Ziel exakt trifft, also genau den Flaechenanteil je Kachel ausgibt, gegen den er
trainiert wird. Seine Punkt-AUC ist damit die DECKE der Rasterweite: besser als
das kann bei dieser Kachelzahl kein Modell werden, egal wie gut es lernt.

Die Decke wird auf demselben Weg gerechnet, den die spaetere Auswertung nimmt,
also mit `to_reference` (bilinear) vom Kopfraster auf 224 hoch. Die Fassung ohne
Glaettung (nearest, das Feld so wie es ist) steht als zweite Spalte daneben,
damit sichtbar bleibt, wie viel die Hochrechnung selbst beitraegt.

DIE VORFESTLEGUNG, verdrahtet in AUC_GATE und IOU_GATE
------------------------------------------------------
Gewaehlt wird die GROEBSTE Rasterweite, die BEIDE Tore nimmt:

  Tor 1   Decke der Punkt-AUC (bilinear, ganzes Bild)  >= AUC_GATE = 0.98
  Tor 2   Decke der IoU des harten Ziels               >= IOU_GATE = 0.70

Warum groebstmoeglich: weniger Kacheln heissen mehr Bildinformation je Kachel
und weniger Stellen, an denen sich der Kopf irren kann. Feiner ist nicht
kostenlos.

Warum Tor 1 bei 0.98: die Rasterung darf hoechstens 2 Prozent der messbaren
Trennschaerfe wegnehmen. Das Projekt loest Unterschiede in der Groessenordnung
0.01 bis 0.02 auf (Fold-Streuung der geschichteten AUC 0.015, gepaartes
Intervall etwa 0.01). Ein Deckenverlust unterhalb davon kann keinen Vergleich
kippen; oberhalb koennte er es.

Warum Tor 2 bei 0.70: Phase 5b liest aus demselben Feld echte Kaesten aus und
berichtet Ueberlappung (IoU) und mAP, also die Zahlen des RSNA-Wettbewerbs. Eine
Decke von 0.53 ist als Wettbewerbszahl nicht mehr berichtbar, egal wie gut das
Modell lernt; 0.70 ist die uebliche Schwelle, ab der eine Detektion als Treffer
zaehlt, und damit die naheliegende Grenze.

EHRLICHE ANMERKUNG ZUR ENTSTEHUNG DIESER REGEL
----------------------------------------------
Tor 1 stand vor der ersten gerechneten Zahl. Tor 2 kam NACH der Messung dazu,
und das gehoert genannt statt versteckt.

Der Grund war nicht, dass das Ergebnis missfiel, sondern dass Tor 1 sich als
untauglich erwies: 7x7 erreicht 0.9928 und 14x14 erreicht 0.9989. Der
Unterschied ist 0.0061 und liegt damit UNTER dem, was das Projekt aufloest. Ein
Kriterium, das saettigt, entscheidet nichts; die Regel "die groebste, die
besteht" waere dann keine Messung mehr gewesen, sondern nur noch der
Ausschlagsatz. Tor 2 misst dieselben Rasterungen an der Groesse, die Phase 5b
berichtet, und dort trennen sie deutlich (0.53 gegen 0.74).

Was daraus fuer die Mappe folgt: die Rasterweite ist NICHT durch eine
vorfestgelegte Regel entschieden worden, sondern durch eine nachtraeglich
ergaenzte. Das ist ein schwaecheres Argument als eine echte Vorfestlegung, und es
wird auch als schwaecheres verkauft. Was es rettet, ist, dass Phase 5b samt IoU
und mAP schon in der Roadmap stand, bevor hier irgendetwas gerechnet wurde. Das
Kriterium war also verfuegbar und wurde vergessen, nicht erfunden.

WAS NEBENBEI HERAUSFAELLT, und gebraucht wird
---------------------------------------------
* `tile_pos_rate`: der Anteil positiver Kacheln. Das ist das `pos_weight` je
  Kachel fuer den Kopfverlust, und es wird hier gemessen statt geschaetzt. Die
  Roadmap nennt "rund 11 Prozent", das ist der FLAECHENanteil der Kaesten; ob
  der Kachelanteil derselbe ist, ist eine eigene Frage.
* `iou_hard`: was ein hartes 0/1-Ziel gegenueber dem weichen kostet. Die
  Roadmap legt weiche Ziele bereits fest; diese Spalte belegt, dass die
  Festlegung richtig war, statt sie nur zu behaupten.
* `dazuerfunden` und `verloren`: die beiden Fehlerrichtungen der harten
  Rasterung, in Anteilen der echten Kastenflaeche.

Gemessen wird auf dem TRAININGSTEIL von Fold 0. Kastengeometrie ist zwar keine
Groesse, die aus der Validierung lecken kann, aber die Regel "Entscheidungen nur
aus dem Trainingsteil" wird hier nicht ausgenommen, damit sie nirgends
diskutierbar wird.

CLI, aus dem Repo-Wurzelverzeichnis:
  .\\venv\\Scripts\\python.exe rsna\\befunde\\rsna_kopfraster.py
  .\\venv\\Scripts\\python.exe rsna\\befunde\\rsna_kopfraster.py --grids 7 14 28
  .\\venv\\Scripts\\python.exe rsna\\befunde\\rsna_kopfraster.py --masks data\\rsna\\masks224_dev
"""

from __future__ import annotations

import _repo_path  # noqa: F401  (legt die Nachbarordner auf sys.path)

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_lokalisation import (BOX_SPACE, REF_SIZE, box_mask, load_boxes,
                               load_lung, point_auc, to_reference)

AUC_GATE = 0.98           # Tor 1, stand vor der ersten Zahl
IOU_GATE = 0.70           # Tor 2, nachtraeglich ergaenzt, Begruendung im Kopf
DEFAULT_GRIDS = (7, 14)


# --------------------------------------------------------------------------
# Das Ziel: der Flaechenanteil je Kachel
# --------------------------------------------------------------------------

def soft_target(boxes, grid: int, size: int = REF_SIZE,
                box_space: int = BOX_SPACE) -> np.ndarray:
    """Der Anteil jeder Kachel, der von einem Kasten bedeckt ist. grid x grid.

    Gerechnet wird ueber die Kastenmaske auf dem Referenzraster und nicht
    analytisch aus den Kastenkoordinaten. Das ist Absicht: so entsteht das Ziel
    aus GENAU derselben Maske, gegen die spaeter gemessen wird, samt derselben
    Abschneidung in `box_mask`. Eine zweite, analytisch saubere Rechnung waere
    genauer und wuerde ein Ziel erzeugen, das die Auswertung gar nicht kennt.

    `size` muss durch `grid` teilbar sein; 224 ist durch 7, 14, 28 und 32
    teilbar. Ist es das nicht, ist das ein harter Abbruch: eine krumme Teilung
    wuerde die Randkacheln stillschweigend kleiner machen und damit den
    Flaechenanteil dort verfaelschen.
    """
    if size % grid:
        raise ValueError(f"Referenzraster {size} ist nicht durch {grid} teilbar.")
    m = box_mask(boxes, size, box_space).astype(np.float32)
    k = size // grid
    return m.reshape(grid, k, grid, k).mean(axis=(1, 3))


def to_field_pixels(field: np.ndarray, size: int = REF_SIZE) -> np.ndarray:
    """Das Kopffeld als Bild, ohne Glaettung: jede Kachel ein Block.

    Das ist das Feld, wie es wirklich ist. `to_reference` glaettet bilinear und
    ist der Weg, den die spaetere Auswertung nimmt. Beide werden berichtet,
    damit der Beitrag der Glaettung sichtbar bleibt und nicht als Eigenschaft
    des Rasters missverstanden wird.
    """
    g = field.shape[0]
    return np.kron(np.asarray(field, dtype=np.float32),
                   np.ones((size // g, size // g), np.float32))


# --------------------------------------------------------------------------
# Die Messung je Bild
# --------------------------------------------------------------------------

def measure_image(boxes, grid: int, lung: np.ndarray | None = None) -> dict:
    """Decke und Rasterfehler fuer EIN Bild bei EINER Rasterweite."""
    truth = box_mask(boxes)                       # 224 x 224, bool
    field = soft_target(boxes, grid)              # grid x grid, weich
    blocky = to_field_pixels(field)               # ohne Glaettung
    smooth = to_reference(field)                  # der Auswertungsweg, bilinear

    hard = (field >= 0.5).astype(np.float32)      # das verworfene 0/1-Ziel
    hard_px = to_field_pixels(hard).astype(bool)

    inter = float((hard_px & truth).sum())
    union = float((hard_px | truth).sum())
    n_true = float(truth.sum())

    out = {
        "grid": grid,
        "ceiling_auc": point_auc(smooth, truth),
        "ceiling_auc_blocky": point_auc(blocky, truth),
        "ceiling_auc_lung": float("nan"),
        "iou_hard": inter / union if union else float("nan"),
        # Beide Fehlerrichtungen getrennt, in Anteilen der echten Kastenflaeche.
        # Zusammengefasst waeren sie wertlos: ein Raster, das gleich viel
        # dazuerfindet wie es verliert, sieht dann fehlerfrei aus.
        "dazuerfunden": float((hard_px & ~truth).sum()) / n_true if n_true else float("nan"),
        "verloren": float((~hard_px & truth).sum()) / n_true if n_true else float("nan"),
        "tile_pos_rate_hard": float(hard.mean()),
        "tile_cov_mean": float(field.mean()),
        "box_area": float(truth.mean()),
    }
    if lung is not None:
        out["ceiling_auc_lung"] = point_auc(smooth, truth, lung)
    return out


# --------------------------------------------------------------------------

def run(args) -> int:
    sp = json.loads(Path(args.splits).read_text())
    boxes = load_boxes(args.csv)
    train_ids = sp["folds"][args.fold]["train"]
    pos = [i for i in train_ids if i in boxes]
    if args.limit:
        pos = pos[:args.limit]
    if not pos:
        print("kein Bild mit Kaesten, nichts zu messen")
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRasterweiten {list(args.grids)} auf {len(pos)} annotierten Bildern "
          f"des Trainingsteils von Fold {args.fold}")
    print(f"Referenzraster {REF_SIZE}, Kaesten aus dem {BOX_SPACE}er DICOM-Raster")
    if args.masks:
        print(f"Lungenmasken aus {args.masks}")

    rows, no_mask = [], 0
    for j, pid in enumerate(pos, 1):
        lung = None
        if args.masks:
            lung = load_lung(args.masks, pid)
            if lung is None:
                no_mask += 1
        for g in args.grids:
            r = measure_image(boxes[pid], g, lung)
            r["patientId"] = pid
            rows.append(r)
        if j % 500 == 0:
            print(f"    {j}/{len(pos)}")
    if no_mask:
        print(f"  {no_mask} Bilder ohne Lungenmaske, dort bleibt die "
              f"Lungenfassung leer")

    d = pd.DataFrame(rows)
    d.to_csv(out_dir / "kopfraster_per_image.csv", index=False)

    # `patientId` fliegt vor der Aggregation raus: pandas versucht sonst den
    # Mittelwert einer Zeichenkette zu bilden und bricht ab.
    s = d.drop(columns=["patientId"]).groupby("grid").agg(["mean", "std"])
    tab = pd.DataFrame({
        "n": d.groupby("grid").size(),
        "Decke bilinear": s[("ceiling_auc", "mean")],
        "sd": s[("ceiling_auc", "std")],
        "Decke ungeglaettet": s[("ceiling_auc_blocky", "mean")],
        "Decke in der Lunge": s[("ceiling_auc_lung", "mean")],
        "IoU hartes Ziel": s[("iou_hard", "mean")],
        "dazuerfunden": s[("dazuerfunden", "mean")],
        "verloren": s[("verloren", "mean")],
        "Kachel positiv hart": s[("tile_pos_rate_hard", "mean")],
        "Kachel Bedeckung weich": s[("tile_cov_mean", "mean")],
    })
    tab.to_csv(out_dir / "kopfraster_summary.csv")

    print("\n" + "=" * 78)
    print("DIE DECKE JE RASTERWEITE, Zufallswert der Punkt-AUC ist 0.5")
    print("=" * 78)
    print(tab.round(4).to_string())

    print(f"\n  Kastenflaeche im Mittel {d['box_area'].mean():.4f} des Bildes.")
    print("  'dazuerfunden' und 'verloren' sind Anteile DIESER Flaeche und")
    print("  gelten nur fuer das verworfene harte 0/1-Ziel. Das weiche Ziel hat")
    print("  diesen Fehler nicht, es traegt den Flaechenanteil selbst.")

    # ---- Die Vorfestlegung anwenden ------------------------------------
    print("\n" + "=" * 78)
    print("ENTSCHEIDUNG nach der Vorfestlegung im Dateikopf")
    print("=" * 78)
    print(f"  Regel: die GROEBSTE Rasterweite mit Punkt-AUC-Decke >= {AUC_GATE:.2f}")
    print(f"         UND IoU-Decke des harten Ziels >= {IOU_GATE:.2f}")
    ok = []
    for g in sorted(args.grids):
        a = float(tab.loc[g, "Decke bilinear"])
        i = float(tab.loc[g, "IoU hartes Ziel"])
        t1 = "erfuellt" if a >= AUC_GATE else "verfehlt"
        t2 = "erfuellt" if i >= IOU_GATE else "verfehlt"
        if a >= AUC_GATE and i >= IOU_GATE:
            ok.append(g)
        print(f"    {g:>3} x {g:<3}  Punkt-AUC {a:.4f} {t1}   "
              f"IoU {i:.4f} {t2}")

    # Wenn Tor 1 nicht trennt, ist das selbst berichtenswert. Ein saettigendes
    # Kriterium sieht wie eine Entscheidung aus und ist keine.
    span = (float(tab["Decke bilinear"].max()) - float(tab["Decke bilinear"].min()))
    if len(args.grids) > 1 and span < 0.01:
        print(f"\n  HINWEIS: Tor 1 trennt nicht. Zwischen der besten und der")
        print(f"  schlechtesten Rasterweite liegen {span:.4f} Punkt-AUC, weniger")
        print("  als die 0.01, die dieses Projekt aufloest. Die Rasterweite ist")
        print("  auf Endpunkt B praktisch gleichgueltig; entschieden hat Tor 2.")
        print("  Genau deshalb steht Tor 2 in der Datei, siehe Dateikopf.")

    if not ok:
        print(f"\n  KEINE Rasterweite nimmt beide Tore.")
        print("  Das ist selbst ein Befund: bei diesen Kastengroessen ist ein")
        print("  Kachelfeld als Ziel grundsaetzlich zu grob. Vor dem Bau klaeren,")
        print("  nicht das Tor senken.")
        return 1
    pick = min(ok)
    print(f"\n  gewaehlt: {pick} x {pick}")
    print(f"  Der Kopf gibt ab jetzt ein {pick} x {pick} Feld aus, und zwar")
    print("  UNABHAENGIG von der Bildgroesse: er greift die Merkmalskarte ab und")
    print("  mittelt sie auf genau diese Kachelzahl herunter. Bei 224 Punkten")
    print("  passt sie ohnehin, bei 512 wird zusammengefasst. Damit aendert sich")
    print("  das Lineal nicht mehr mit dem Gemessenen, und der")
    print("  Aufloesungsvergleich in Phase 8 wird zum ersten Mal sauber.")

    pw = float(tab.loc[pick, "Kachel Bedeckung weich"])
    print(f"\n  pos_weight je Kachel bei {pick} x {pick}: "
          f"{(1 - pw) / pw:.2f}")
    print(f"  (mittlere Bedeckung {pw:.4f}, also "
          f"(1 - {pw:.4f}) / {pw:.4f}, dieselbe Rechnung wie das pos_weight")
    print("  der Klassifikation, nur je Kachel statt je Bild)")

    print(f"\ngespeichert: {out_dir}/kopfraster_per_image.csv, "
          f"{out_dir}/kopfraster_summary.csv")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--masks", type=Path, default=None,
                   help="Lungenmasken, z. B. data/rsna/masks224_dev. Ohne diese "
                        "Angabe bleibt die Lungenfassung der Decke leer; die "
                        "Entscheidung haengt nicht an ihr")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--grids", type=int, nargs="+", default=list(DEFAULT_GRIDS))
    p.add_argument("--limit", type=int, default=0,
                   help="nur die ersten n Bilder, 0 = alle")
    p.add_argument("--out-dir", type=Path, default=Path("predictions_kopf"))
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
