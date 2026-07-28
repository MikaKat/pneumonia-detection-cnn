"""
Schritt 9c: War der Spielraum ein Maskenartefakt? -- Varianten in Sekunden.

Warum es dieses Skript gibt
---------------------------
Der erste Diagnoselauf hat zwei Dinge gleichzeitig ergeben:

  * Der Aussen-Befund traegt nicht. "80,7 % der Fehlschlaege zeigen aus der
    Lunge" steht gegen einen Zufallswert von 79,2 % -- Vorsprung +0,014 bei
    t = 0,32. Nichts.
  * Trotzdem hebt die Beschraenkung auf die Lunge die Trefferquote um +0,080,
    und zwar in 5 von 5 Folds.

Der zweite Punkt haengt aber an einer Maske, die messbar zu klein ist:
Lungenflaeche 0,210 statt der anatomisch erwartbaren 0,30-0,40, und 28,5 % der
Bounding-Box-Flaeche liegen ausserhalb. Eine zu kleine Maske erzeugt beides von
selbst -- sie nimmt einem Rausch-Maximum Platz weg, ganz ohne Modellverdienst.

Die saubere Gegenprobe ist deshalb: Maske schrittweise groesser machen und
zusehen, was mit dem Spielraum passiert.

  * Schrumpft der Spielraum gegen den geometrischen Nullwert, sobald die Maske
    anatomisch plausibel ist -> er war ein Artefakt. Crop erledigt, 11,5 h
    gespart, und das Ergebnis ist eine Aussage, kein Ausfall.
  * Bleibt er bestehen -> das Sichtfeld ist tatsaechlich das Problem, und der
    Crop-Lauf ist begruendet.

Warum das jetzt Sekunden kostet
-------------------------------
Teuer sind zwei Dinge: der U-Net-Forward-Pass (15 min) und die Grad-CAM-
Rechnung (20 min). Beide haengen NICHT von der Verfeinerung ab. Also werden
beide einmal gecacht -- die rohe U-Net-Ausgabe als gepackte Bits
(`rsna_make_masks.py --raw-cache`), die Heatmaps als float16
(`rsna_cam_lung_check.py --cache-heat`). Jede weitere Variante ist danach
Morphologie plus ein paar argmax-Aufrufe.

Es werden bewusst KEINE Masken auf die Platte geschrieben. Ein Sweep, der
nebenbei den Maskenordner ueberschreibt, waere eine stille Falle fuer den
naechsten Lauf.

CLI:
  # Voraussetzung (je einmal):
  #   python rsna_make_masks.py --ids-from "predictions_rsna/cam_f*_s0.csv" \
  #       --raw-cache data/rsna/unet_raw256.npz
  #   python rsna_cam_lung_check.py --folds 0 1 2 3 4 --n 120 --cache-heat

  python rsna_mask_sweep.py
  python rsna_mask_sweep.py --variants none:0 default:0 default:6 hull:0 hull:6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_cam_lung_check import analyse_one, box_mask, cv_mean, paired_t, summarise
from rsna_make_masks import refine_variant, to_out, unpack_masks

DEFAULT_VARIANTS = ["none:0", "default:0", "default:4", "default:8",
                    "hull:0", "hull:4", "hull:8"]

BOX_SPACE = 1024        # Bounding Boxes liegen im Original-DICOM-Raster


def load_boxes(csv_dir: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    """Bewusste Kopie von `rsna_train.load_boxes`.

    Der Sweep ist reine Nachbearbeitung auf zwei Caches -- er rechnet kein
    Modell und braucht kein Torch. `rsna_train` importiert Torch aber auf
    Modulebene, sodass ein Import von dort den Sweep an eine schwere und hier
    voellig unnoetige Abhaengigkeit kettet (und ihn auf jeder Maschine ohne
    GPU-Stack unbrauchbar macht).

    Gegen Auseinanderdriften der beiden Fassungen sichert
    `test_rsna_masks.test_load_boxes_matches_train` ab: ist Torch vorhanden,
    wird gegen das Original verglichen.
    """
    df = pd.read_csv(Path(csv_dir) / "stage_2_train_labels.csv")
    df = df[df["Target"] == 1].dropna(subset=["x", "y", "width", "height"])
    out: dict[str, list] = {}
    for pid, x, y, w, h in df[["patientId", "x", "y", "width", "height"]].values:
        out.setdefault(pid, []).append((float(x), float(y), float(w), float(h)))
    return out


def parse_variant(s: str) -> tuple[str, int]:
    """'hull:4' -> ('hull', 4). Eigene Funktion, weil ein stiller Parse-Fehler
    hier die ganze Tabelle unbrauchbar machte, ohne aufzufallen."""
    if ":" not in s:
        return s, 0
    mode, _, px = s.partition(":")
    return mode, int(px)


def load_heat_cache(pred_dir: Path, fold: int, seed: int):
    p = Path(pred_dir) / f"cam_heat_f{fold}_s{seed}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return [str(s) for s in z["ids"]], z["heat"]


def run_variant(mode: str, dilate: int, raw_lookup: dict[str, np.ndarray],
                heat_by_fold: dict[int, tuple[list[str], np.ndarray]],
                boxes: dict, size: int, box_space: int) -> list[dict]:
    """Eine Verfeinerungs-Einstellung ueber alle Folds auswerten."""
    per_fold = []
    for fold, (ids, heats) in sorted(heat_by_fold.items()):
        rows = []
        for pid, heat in zip(ids, heats):
            raw = raw_lookup.get(pid)
            if raw is None:
                continue
            lung = to_out(refine_variant(raw, mode, dilate), size) > 127
            r = analyse_one(np.asarray(heat, dtype=np.float32),
                            box_mask(boxes.get(pid, []), size, box_space), lung)
            if r is not None:
                rows.append(r)
        if not rows:
            continue
        s = summarise(pd.DataFrame(rows))
        s["fold"] = fold
        per_fold.append(s)
    return per_fold


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-cache", type=Path, default=Path("data/rsna/unet_raw256.npz"))
    p.add_argument("--pred-dir", type=Path, default=Path("predictions_rsna"))
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                   help="modus:dilatation, z.B. hull:4")
    p.add_argument("--out", type=Path, default=Path("predictions_rsna/mask_sweep.csv"))
    args = p.parse_args(argv)

    if not args.raw_cache.exists():
        print(f"FEHLER: Roh-Cache fehlt: {args.raw_cache}")
        print('  python rsna_make_masks.py --ids-from "predictions_rsna/cam_f*_s0.csv" '
              f'--raw-cache {args.raw_cache}')
        return 2

    z = np.load(args.raw_cache, allow_pickle=False)
    raw_ids = [str(s) for s in z["ids"]]
    raw_lookup = {pid: unpack_masks(z["packed"][i:i + 1])[0]
                  for i, pid in enumerate(raw_ids)}
    print(f"Roh-Cache: {len(raw_lookup)} U-Net-Ausgaben")

    heat_by_fold = {}
    for f in args.folds:
        got = load_heat_cache(args.pred_dir, f, args.seed)
        if got is None:
            print(f"  Fold {f}: kein Heat-Cache -- uebersprungen.")
            continue
        heat_by_fold[f] = got
    if not heat_by_fold:
        print("FEHLER: kein Heat-Cache gefunden.")
        print("  python rsna_cam_lung_check.py --folds 0 1 2 3 4 --n 120 --cache-heat")
        return 2
    print(f"Heat-Cache: {sum(len(v[0]) for v in heat_by_fold.values())} Karten "
          f"aus {len(heat_by_fold)} Folds\n")

    boxes = load_boxes(args.csv)

    rows = []
    for v in args.variants:
        mode, dil = parse_variant(v)
        pf = run_variant(mode, dil, raw_lookup, heat_by_fold, boxes,
                         args.size, BOX_SPACE)
        if not pf:
            continue
        rec = {"variante": v}
        for k in ("lung_area", "box_in_lung", "peak_in_box", "peak_in_lung",
                  "miss_outside_lung", "miss_outside_null", "miss_outside_lift",
                  "crop_headroom", "headroom_null", "headroom_vs_null",
                  "lift_free", "lift_restricted", "lift_delta"):
            m, s = cv_mean(pf, k)
            rec[k] = m
            rec[k + "_sd"] = s
        rec["t_miss"] = paired_t(pf, "miss_outside_lift")
        rec["t_headroom_vs_null"] = paired_t(pf, "headroom_vs_null")
        rows.append(rec)
        print(f"  {v:<12} Lungenflaeche {rec['lung_area']:.3f}  "
              f"box_in_lung {rec['box_in_lung']:.3f}  "
              f"Spielraum {rec['crop_headroom']:+.3f}")

    if not rows:
        print("Keine Variante auswertbar.")
        return 1

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # ---- Tabelle ----------------------------------------------------------
    print("\n" + "=" * 96)
    print("MASKENVARIANTEN -- was aendert sich, wenn die Maske plausibel wird?")
    print("=" * 96)
    print(f"{'Variante':<12}{'Lunge':>8}{'boxInL':>8}{'aussen':>9}{'Zufall':>9}"
          f"{'Vorspr':>9}{'VorFrei':>10}{'VorBeschr':>11}{'DIFF':>8}{'t':>8}")
    print("-" * 96)
    for _, r in df.iterrows():
        print(f"{r['variante']:<12}{r['lung_area']:>8.3f}{r['box_in_lung']:>8.3f}"
              f"{r['miss_outside_lung']:>9.3f}{r['miss_outside_null']:>9.3f}"
              f"{r['miss_outside_lift']:>+9.3f}{r['lift_free']:>+10.3f}"
              f"{r['lift_restricted']:>+11.3f}{r['lift_delta']:>+8.3f}"
              f"{r['t_headroom_vs_null']:>8.2f}")
    print("-" * 96)
    print("  Lunge     = Maskenflaeche (anatomisch ~0,30-0,40 erwartbar)")
    print("  boxInL    = Anteil der Bounding Box in der Maske (Kontrolle gegen")
    print("              Untersegmentierung; niedrig = Maske schneidet Pathologie weg)")
    print("  aussen    = Fehlschlaege mit Max ausserhalb der Lunge, Zufall = 1-Lunge")
    print("  VorFrei   = Trefferquote MINUS Boxflaeche")
    print("  VorBeschr = Trefferquote(nur Lunge) MINUS (Box UND Lunge)/Lunge")
    print("  DIFF      = VorBeschr - VorFrei, gepaart je Fold. Beide sind Abstaende")
    print("              zur je EIGENEN Null, also deckeneffekt-fest vergleichbar.")
    print("              Negativ = die Lungen-Beschraenkung macht das Maximum")
    print("              weniger informativ, nicht mehr.")

    # ---- Lesart -----------------------------------------------------------
    ok = df[(df.lung_area >= 0.26) & (df.box_in_lung >= 0.60)]
    print("\n" + "=" * 96)
    print("LESART")
    print("=" * 96)
    if ok.empty:
        print("  KEINE Variante besteht beide Maskenkontrollen (Lungenflaeche >= 0,26")
        print("  UND box_in_lung >= 0,60). Das U-Net untersegmentiert auf RSNA staerker,")
        print("  als sich mit Huelle und Dilatation beheben laesst.")
        print("  -> Dann ist nicht der Crop die naechste Frage, sondern ob die")
        print("     Segmentierung in diesem Projekt ueberhaupt tragfaehig ist.")
        print("     Das ist ein berichtenswertes Ergebnis, kein Ausfall.")
    else:
        best = ok.loc[ok.headroom_vs_null.idxmax()]
        print(f"  Beste Variante, die beide Kontrollen besteht: {best['variante']}")
        print(f"    Lungenflaeche {best['lung_area']:.3f} | "
              f"box_in_lung {best['box_in_lung']:.3f}")
        print(f"    Vorsprung frei {best['lift_free']:+.3f} gegen beschraenkt "
              f"{best['lift_restricted']:+.3f}  ->  Differenz "
              f"{best['lift_delta']:+.3f} (t = {best['t_headroom_vs_null']:+.2f})")
        print(f"    Aussen-Vorsprung {best['miss_outside_lift']:+.3f} "
              f"(t = {best['t_miss']:+.2f})")
        if best["headroom_vs_null"] > 0.02 and abs(best["t_miss"]) > 2.78:
            print("  -> Beides traegt: der Crop-Lauf ist begruendet. GEPAART je Fold")
            print("     messen, sonst bedeutet eine Differenz von 0,005 nichts.")
        else:
            print("  -> Auch mit plausibler Maske bleibt kein Ueberschuss ueber die")
            print("     reine Flaechenrechnung. Der Spielraum war Geometrie, nicht")
            print("     Sichtfeld. Schritt 3 abwaehlen -- 11,5 h gespart, und der")
            print("     Befund 'Segmentierung traegt hier nichts' gehoert so in die Mappe.")
    print("=" * 96)
    print(f"\nTabelle: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
