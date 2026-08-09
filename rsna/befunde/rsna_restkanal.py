"""What geometry is still IN THE IMAGE after the fixed crop?

The question this answers, and why the existing table does not
------------------------------------------------------------------
`qc/crop_varianten_tabelle.csv` reports `AUC_Geometrie_zu_APPA` per crop
variant: 0.714 for the adaptive window, 0.554 for the fixed one. Those numbers
describe the geometry OF THE CROP WINDOW. That is the right question for the
failure of 26.07: there the window itself was the leak, because its size was
fitted to the lung and the lung looks smaller on supine films.

It is not the question a trained model faces. The model never sees the window
parameters. It sees pixels, and the pixels still contain a lung. How much of
the projection can be read from the lung's own rectangle, in the coordinate
frame of the crop, is a different measurement, and nothing in this repository
had made it.

Same instrument, three frames
-----------------------------
Every number here comes from `rsna_crop_geometry.cv_auc` on the same ten
rectangle features, so this is a change of frame and not a change of
instrument. Two of the three answers are already known, and the script refuses
to report the third unless it reproduces both:

  adaptive window   has to come out near 0.7144   (table row "IST: hull d8 pad.05")
  fixed window 0.80 has to come out near 0.5545   (table row "FIX .80")
  lung rectangle inside the crop                  <- the new number

A calibration that fails means the rebuild is not the original measurement, and
then the new number means nothing either. See the memory note on changing the
instrument after seeing the result.

Reading the output
------------------
The comparison that matters is "what can be read off the whole image" against
"what can be read off the crop", both as distance from the coin, `|AUC - 0.5|`.
The share removed is what the crop actually takes away from the model, as
opposed to what it takes away from the window parameters.

A residual well above zero does not invalidate phase 7. It bounds what a null
result may be read to mean: not "geometry is exonerated" but "the FRAMING is
exonerated". Anatomical size in a fixed frame is not framing, and it cannot be
cropped away without cropping away the finding.

CLI:
  python rsna\\befunde\\rsna_restkanal.py
  python rsna\\befunde\\rsna_restkanal.py --params predictions_rsna/crop_params_fix080.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _repo_path  # noqa: F401  (puts the neighbouring folders on sys.path)

from rsna_crop_geometry import FEATURES, cv_auc
from rsna_make_crops import square_crop

PAD = 0.05
REFINE, DILATE = "hull", 8

# The two numbers the rebuild has to reproduce, from qc/crop_varianten_tabelle.csv.
EICHUNG = {"adaptives Fenster": 0.7144, "festes Fenster": 0.5545}
EICH_TOLERANZ = 0.010


def merkmale(y0: float, y1: float, x0: float, x1: float) -> dict:
    """The ten features of `rsna_crop_geometry.rect_features`, from a rectangle
    that is already normalised, so the same vector can describe a rectangle in
    the whole image and in the crop."""
    h, w = y1 - y0, x1 - x0
    return {"y0": y0, "y1": y1, "x0": x0, "x1": x1, "height": h, "width": w,
            "aspect": (w / h) if h > 0 else np.nan,
            "cy": (y0 + y1) / 2, "cx": (x0 + x1) / 2, "area": h * w}


def rechtecke_laden(cache: Path, ids: list[str], zwischen: Path) -> dict:
    """Bounding rectangle of the refined lung mask, per image.

    Cached, because refining 22872 masks costs minutes and this script is
    meant to be rerunnable while a training run holds the graphics card.
    """
    if zwischen.exists():
        d = np.load(zwischen, allow_pickle=False)
        got = {str(p): b for p, b in zip(d["patientId"], d["bbox"])}
        if set(got) >= set(ids):
            print(f"  Rechtecke aus {zwischen} ({len(got)})")
            return got
        print(f"  {zwischen} deckt die Auswahl nicht ab, wird neu gebaut.")

    from rsna_make_crops import mask_bbox
    from rsna_make_masks import load_raw_cache, refine_variant, unpack_masks

    cache_ids, packed = load_raw_cache(cache)
    index = {p: i for i, p in enumerate(cache_ids)}
    out, t0 = {}, time.time()
    for k, pid in enumerate(ids, 1):
        if pid not in index:
            continue
        raw = unpack_masks(packed[index[pid]:index[pid] + 1])[0]
        bb = mask_bbox(refine_variant(raw, REFINE, DILATE))
        if bb is not None:
            out[pid] = np.asarray(bb, dtype=float)
        if k % 4000 == 0 or k == len(ids):
            print(f"    Maske {k}/{len(ids)}  ({(time.time() - t0) / 60:.1f} min)",
                  flush=True)
    zwischen.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(zwischen, patientId=np.array(list(out)),
                        bbox=np.array(list(out.values())))
    print(f"  Rechtecke gespeichert: {zwischen}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", type=Path,
                   default=Path("predictions_rsna/crop_params_fix080.csv"))
    p.add_argument("--raw-cache", type=Path,
                   default=Path("data/rsna/unet_raw256.npz"))
    p.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    p.add_argument("--zwischen", type=Path,
                   default=Path("predictions_rsna/lungenrechtecke.npz"))
    p.add_argument("--out", type=Path,
                   default=Path("predictions_rsna/restkanal.csv"))
    args = p.parse_args(argv)

    for f in (args.params, args.raw_cache, args.splits):
        if not f.exists():
            print(f"ABBRUCH: {f} fehlt.")
            return 2

    par = pd.read_csv(args.params)
    par["patientId"] = par["patientId"].astype(str)
    vp = json.loads(args.splits.read_text())["viewpos"]
    par["vp"] = par["patientId"].map(vp)
    par = par[par["vp"].isin(["AP", "PA"])].reset_index(drop=True)
    print(f"Bilder: {len(par)}, davon AP {(par['vp'] == 'AP').sum()}")

    bbs = rechtecke_laden(args.raw_cache, list(par["patientId"]), args.zwischen)
    par = par[par["patientId"].isin(bbs)].reset_index(drop=True)
    y = (par["vp"] == "AP").to_numpy().astype(int)

    ganz, adaptiv, fest, zugeschnitten = [], [], [], []
    for pid, top, left, side in par[["patientId", "top", "left",
                                     "side"]].itertuples(index=False):
        t, l, b, r = bbs[pid]
        ph, pw = PAD * (b - t), PAD * (r - l)
        gy0, gy1 = max(t - ph, 0.0), min(b + ph, 1.0)
        gx0, gx1 = max(l - pw, 0.0), min(r + pw, 1.0)
        ganz.append(merkmale(gy0, gy1, gx0, gx1))

        at, al, asd = square_crop((t, l, b, r), PAD)
        adaptiv.append(merkmale(at, at + asd, al, al + asd))
        fest.append(merkmale(top, top + side, left, left + side))

        # The same lung rectangle, clipped to the window and expressed in
        # units of the window. That is what the pixels of the crop show.
        zugeschnitten.append(merkmale(
            (max(gy0, top) - top) / side, (min(gy1, top + side) - top) / side,
            (max(gx0, left) - left) / side, (min(gx1, left + side) - left) / side))

    def auc(rows) -> float:
        X = pd.DataFrame(rows)[FEATURES].to_numpy(float)
        ok = np.isfinite(X).all(axis=1)
        return cv_auc(X[ok], y[ok])

    print("\n" + "=" * 74)
    print("EICHUNG an zwei Zahlen, die feststehen")
    print("=" * 74)
    print("  qc/crop_varianten_tabelle.csv misst die Geometrie DES FENSTERS.")
    print("  Reproduziert der Nachbau beide Zeilen, ist er dasselbe Messgeraet.")
    print()
    werte = {"adaptives Fenster": auc(adaptiv), "festes Fenster": auc(fest)}
    gut = True
    for name, soll in EICHUNG.items():
        ist = werte[name]
        ab = abs(ist - soll)
        zeichen = "ok  " if ab <= EICH_TOLERANZ else "FEHLER"
        print(f"  {zeichen} {name:<22} {ist:.4f}   erwartet {soll:.4f}   "
              f"Abweichung {ab:.4f}")
        gut &= ab <= EICH_TOLERANZ
    if not gut:
        print("\nABBRUCH: der Nachbau reproduziert die bekannten Zahlen nicht.")
        print("Dann ist es ein anderes Messgeraet, und die Zahl unten waere")
        print("mit der Tabelle nicht vergleichbar.")
        return 1

    a_ganz, a_zu = auc(ganz), auc(zugeschnitten)
    print("\n" + "=" * 74)
    print("WAS DAS NETZ SIEHT, vorher und nachher")
    print("=" * 74)
    print("  Das Modell bekommt nie die Fensterparameter. Es bekommt Bildpunkte,")
    print("  und darin liegt eine Lunge. Gemessen wird deren eigenes Rechteck,")
    print("  einmal im Rahmen des ganzen Bildes und einmal im Rahmen des")
    print("  Zuschnitts.")
    print()
    print(f"  {'Rahmen':<40}{'AUC':>9}{'Abstand':>10}")
    print(f"  {'ganzes Bild (wie Phase 5 trainiert)':<40}{a_ganz:>9.4f}"
          f"{abs(a_ganz - 0.5):>10.4f}")
    print(f"  {'Zuschnitt (dieser Arm)':<40}{a_zu:>9.4f}"
          f"{abs(a_zu - 0.5):>10.4f}")
    anteil = 100 * (1 - abs(a_zu - 0.5) / abs(a_ganz - 0.5))
    print(f"\n  Der Zuschnitt nimmt {anteil:.0f} Prozent davon.")
    print()
    print("  Zum Vergleich, aus der Tabelle und oben nachgerechnet: von der")
    print(f"  Geometrie DES FENSTERS nimmt er "
          f"{100 * (1 - abs(werte['festes Fenster'] - 0.5) / abs(werte['adaptives Fenster'] - 0.5)):.0f} "
          f"Prozent. Die beiden Zahlen")
    print("  widersprechen sich nicht, sie beantworten verschiedene Fragen.")

    print("\n" + "=" * 74)
    print("EINZELMERKMALE im Zuschnitt, absteigend")
    print("=" * 74)
    Z = pd.DataFrame(zugeschnitten)
    einzeln = []
    for f in FEATURES:
        X = Z[[f]].to_numpy(float)
        ok = np.isfinite(X).all(axis=1)
        a = cv_auc(X[ok], y[ok])
        einzeln.append({"merkmal": f, "auc": a, "abstand": abs(a - 0.5)})
    for r in sorted(einzeln, key=lambda r: -r["abstand"])[:5]:
        print(f"  {r['merkmal']:<10} AUC {r['auc']:.4f}   "
              f"Abstand {r['abstand']:.4f}")

    print("\n" + "=" * 74)
    print("WAS DARAUS FOLGT, UND WAS NICHT")
    print("=" * 74)
    print("  NICHT: dass Phase 7 falsch angelegt ist. Der Endpunkt C misst den")
    print("  Kanal am fertigen Modell und nicht an dieser Rechnung.")
    print()
    print("  WOHL: dass ein Nullergebnis auf C NICHT heissen darf 'die")
    print("  Geometrie ist entlastet'. Entlastet waere die RAHMUNG. Die")
    print("  anatomische Groesse der Lunge in einem festen Rahmen ist keine")
    print("  Rahmung, und sie laesst sich nicht wegschneiden, ohne den Befund")
    print("  mit wegzuschneiden.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"rahmen": "ganzes Bild", "auc": a_ganz, "abstand": abs(a_ganz - 0.5)},
        {"rahmen": "Zuschnitt", "auc": a_zu, "abstand": abs(a_zu - 0.5)},
        {"rahmen": "adaptives Fenster", "auc": werte["adaptives Fenster"],
         "abstand": abs(werte["adaptives Fenster"] - 0.5)},
        {"rahmen": "festes Fenster", "auc": werte["festes Fenster"],
         "abstand": abs(werte["festes Fenster"] - 0.5)},
    ] + einzeln).to_csv(args.out, index=False)
    print(f"\ngespeichert: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
