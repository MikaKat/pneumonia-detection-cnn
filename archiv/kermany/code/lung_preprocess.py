"""
Lungenmasken reparieren und drei Preprocessing-Varianten erzeugen.

Varianten (alle mit IDENTISCHEM Crop, damit der Vergleich nur den
Maskierungs-Effekt misst und nicht den Crop-Effekt):

  crop      : Bounding-Box-Crop, alle Pixel bleiben erhalten
  softmask  : Crop + weiche Abschwaechung ausserhalb der Lunge
  hardmask  : Crop + hartes Ausschneiden (dein bisheriges Verfahren)

Zusaetzlich wird protokolliert, WIE stark jedes Bild gezoomt wurde
(crop_log.csv). Das ist die entscheidende Kontrolle: wenn die Masken bei
einer Klasse systematisch kleiner sind, ist auch der Crop enger, und der
Zoomfaktor wird selbst zum Label-Proxy -- derselbe Confounder wie vorher,
nur an einer anderen Stelle.

CLI:
  python lung_preprocess.py --images data/chest_xray --masks data/chest_xray_masks \
      --out data/prepared --size 512 --save-masks --mirror
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.measure import label, regionprops
from skimage.morphology import closing as _closing, dilation as _dilation, disk
from skimage.transform import resize as sk_resize
from tqdm import tqdm

try:  # skimage >= 0.19
    from skimage.morphology import convex_hull_image
except ImportError:  # aeltere Versionen
    from skimage.morphology.convex_hull import convex_hull_image


# --------------------------------------------------------------------------
# Maskenreparatur
# --------------------------------------------------------------------------

def keep_largest_components(mask: np.ndarray, n: int = 2, min_frac: float = 0.15):
    """Behaelt die n groessten Komponenten. Gibt (maske, n_behalten) zurueck.

    min_frac: Komponenten kleiner als min_frac * groesste Komponente werden
    verworfen (Rauschen, Fragmente).
    """
    lbl = label(mask)
    if lbl.max() == 0:
        return mask.astype(bool), 0
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1]
    biggest = sizes[order[0]]
    keep = [i for i in order[:n] if sizes[i] >= min_frac * biggest]
    return np.isin(lbl, keep), len(keep)


def repair_mask(
    mask: np.ndarray,
    close_radius: int = 6,
    use_convex_hull: bool = True,
    hull_limit_px: int = 10,
    include_mediastinum: bool = False,
    dilate_px: int = 6,
) -> np.ndarray:
    """Repariert eine Lungenmaske, die durch Konsolidierungen ausgefranst ist.

    ACHTUNG zu den Radien: sie gelten im ARBEITSGITTER der Maske (hier 224 px,
    so wie das U-Net sie erzeugt hat), nicht in Originalbild-Pixeln. Die
    Morphologie hier auf voller Aufloesung zu rechnen waere sinnlos -- unterhalb
    des 224er-Gitters gibt es keine Information -- und praktisch unbezahlbar:
    ein disk(60) hat ~11 000 Pixel und laeuft auf 2,4 Mio. Bildpixeln minutenlang.
    Faustregel: 1 px hier entspricht bei einem 1800er Bild etwa 8 px im Original.

    Reihenfolge ist wichtig:
      1. nur die zwei groessten Komponenten (Fragmente raus)
      2. Loecher fuellen + Closing  -> kleine Einbuchtungen
      3. Convex Hull pro Lunge      -> grosse weggeschnittene Verschattungen
      4. optional Mediastinum       -> zeilenweise fuellen
      5. Dilatation                 -> Randverlust ausgleichen

    hull_limit_px begrenzt den Convex Hull auf eine Dilatation der
    Originalmaske. Ohne diese Begrenzung ueberschiesst der Hull bei stark
    gekruemmten Lungen (v.a. Zwerchfellwinkel) deutlich zu weit.
    """
    m = mask.astype(bool)
    m, _ = keep_largest_components(m, n=2)
    if not m.any():
        return m

    m = binary_fill_holes(m)
    if close_radius > 0:
        m = _closing(m, disk(close_radius))

    if use_convex_hull:
        limit = _dilation(m, disk(hull_limit_px)) if hull_limit_px > 0 else None
        hull = np.zeros_like(m)
        lbl = label(m)
        for region in regionprops(lbl):
            sub = lbl[region.slice] == region.label
            hull[region.slice] |= convex_hull_image(sub)
        m = (hull & limit) if limit is not None else hull

    if include_mediastinum:
        rows = np.where(m.any(axis=1))[0]
        for y in rows:
            xs = np.flatnonzero(m[y])
            m[y, xs[0]:xs[-1] + 1] = True

    if dilate_px > 0:
        m = _dilation(m, disk(dilate_px))

    return m


def mirror_missing_lung(mask: np.ndarray, area_ratio: float = 0.45) -> tuple[np.ndarray, bool]:
    """Spiegelt die gute Lunge, falls eine Seite fehlt oder viel zu klein ist.

    Achtung: die Lungen sind nicht symmetrisch (rechts hoeher/breiter,
    links Herzbucht). Gespiegelte Masken sitzen anatomisch leicht falsch --
    deshalb wird das Flag zurueckgegeben, damit du diese Faelle im Test
    getrennt auswerten kannst.
    """
    m = mask.astype(bool)
    lbl = label(m)
    props = sorted(regionprops(lbl), key=lambda r: r.area, reverse=True)
    if len(props) == 0:
        return m, False
    if len(props) >= 2 and props[1].area >= area_ratio * props[0].area:
        return m, False  # beide Lungen plausibel

    good = lbl == props[0].label
    cx_img = m.shape[1] / 2.0
    cx_lung = props[0].centroid[1]
    # an der Bildmitte spiegeln und um die Distanz zur Mitte versetzen
    flipped = np.fliplr(good)
    shift = int(round(2 * (cx_img - cx_lung))) - int(round(2 * (cx_img - (m.shape[1] - 1 - cx_lung))))
    out = m | np.roll(flipped, shift, axis=1) if shift else m | flipped
    return out, True


# --------------------------------------------------------------------------
# Crop + Varianten
# --------------------------------------------------------------------------

def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Enge Bounding-Box der Maske als (y0, x0, y1, x1), y1/x1 exklusiv."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        h, w = mask.shape
        return 0, 0, h, w
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def crop_pad(arr: np.ndarray, top: float, left: float, height: float,
             width: float) -> np.ndarray:
    """Schneidet ein Fenster aus, das ueber den Rand hinausragen DARF.

    Ueberstehende Bereiche werden mit 0 aufgefuellt statt geclippt. Das ist
    der entscheidende Unterschied: clippt man, wird aus dem quadratischen
    Crop ein rechteckiger, und der anschliessende Resize auf ein Quadrat
    streckt das Bild wieder -- genau der Confounder, den der Crop beseitigen
    soll (Seitenverhaeltnis-Leak, AUC 0.84 auf diesem Datensatz).
    """
    t, l = int(round(top)), int(round(left))
    h, w = int(round(height)), int(round(width))
    out = np.zeros((h, w), dtype=arr.dtype)
    y0, x0 = max(0, t), max(0, l)
    y1, x1 = min(arr.shape[0], t + h), min(arr.shape[1], l + w)
    if y1 > y0 and x1 > x0:
        out[y0 - t:y1 - t, x0 - l:x1 - l] = arr[y0:y1, x0:x1]
    return out


def make_variants(
    img_c: np.ndarray,
    mask_c: np.ndarray,
    soft_floor: float = 0.30,
    soft_sigma: float = 10.0,
) -> dict[str, np.ndarray]:
    """Erzeugt crop / softmask / hardmask als uint8-Arrays.

    img_c und mask_c sind bereits gecroppt UND auf die Zielgroesse skaliert.
    Der teure Teil (Resize des Originalbilds) passiert damit genau einmal statt
    dreimal, und der Gaussfilter laeuft auf 512x512 statt auf voller Aufloesung.
    """
    img = img_c.astype(np.float32)
    if img.max() > 1.5:
        img /= 255.0
    m = np.clip(mask_c.astype(np.float32), 0, 1)

    soft = np.clip(gaussian_filter(m, sigma=soft_sigma), 0, 1)
    out = {
        "crop": img,
        "softmask": img * (soft_floor + (1 - soft_floor) * soft),
        "hardmask": img * m,
    }
    return {k: np.clip(v * 255, 0, 255).astype(np.uint8) for k, v in out.items()}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def rank_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Richtungsunabhaengiger AUC ohne sklearn (Mann-Whitney-U auf Raengen)."""
    y = np.asarray(y); s = np.asarray(s, float)
    if len(set(y.tolist())) < 2:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    a = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return max(a, 1 - a)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--masks", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--size", type=int, default=512)
    # Alle Radien gelten im Maskengitter (224), NICHT in Originalbild-Pixeln
    p.add_argument("--dilate", type=int, default=6)
    p.add_argument("--close", type=int, default=6)
    p.add_argument("--hull-limit", type=int, default=10)
    p.add_argument("--mediastinum", action="store_true")
    p.add_argument("--no-hull", action="store_true")
    p.add_argument("--variants", default="crop,softmask,hardmask")
    p.add_argument("--center-frac", type=float, default=0.85,
                   help="Kantenlaenge der Variante 'centercrop' als Anteil der "
                        "kurzen Bildkante -- die Kontrollvariante OHNE Segmentierung")
    p.add_argument("--mirror", action="store_true",
                   help="fehlende/zu kleine Lunge durch Spiegelung ergaenzen")
    p.add_argument("--save-masks", action="store_true",
                   help="reparierte Masken mitschreiben (fuer mask_leakage_check.py)")
    p.add_argument("--margin", type=float, default=0.10)
    args = p.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = sorted(f for f in args.images.rglob("*") if f.suffix.lower() in exts)
    if not files:
        raise SystemExit(f"Keine Bilder unter {args.images}")

    n_ok = n_skip = n_mirrored = 0
    log = []
    for f in tqdm(files, desc="preprocess", unit="img", smoothing=0.05):
        rel = f.relative_to(args.images)  # erhaelt Klassen-Unterordner
        mpath = args.masks / rel
        if not mpath.exists():
            cands = list((args.masks / rel.parent).glob(rel.stem + ".*"))
            if not cands:
                n_skip += 1
                continue
            mpath = cands[0]

        img = load_gray(f)
        # Die Maske bleibt in IHREM Gitter (224). Kein Hochskalieren: die
        # Morphologie waere dort weder genauer noch bezahlbar.
        mask_s = load_gray(mpath) > 127

        # Spiegelung VOR der Reparatur: danach hat die Dilatation die beiden
        # Lungen u.U. zu einer Komponente verschmolzen, dann greift die
        # Symmetrie-Heuristik faelschlich.
        mirrored = False
        if args.mirror:
            mask_s, mirrored = mirror_missing_lung(mask_s)

        mask_s = repair_mask(
            mask_s,
            close_radius=args.close,
            use_convex_hull=not args.no_hull,
            hull_limit_px=args.hull_limit,
            include_mediastinum=args.mediastinum,
            dilate_px=args.dilate,
        )
        if mask_s.sum() < 0.02 * mask_s.size:
            n_skip += 1
            continue

        # Bounding-Box im Maskengitter bestimmen und auf Bildkoordinaten
        # umrechnen. Achtung: sy != sx, weil die 224er-Maske aus einem
        # gestreckten Resize des Originalbilds stammt.
        y0, x0, y1, x1 = mask_bbox(mask_s)
        sy = img.shape[0] / mask_s.shape[0]
        sx = img.shape[1] / mask_s.shape[1]
        box_h, box_w = (y1 - y0) * sy, (x1 - x0) * sx
        cy, cx = (y0 + y1) / 2 * sy, (x0 + x1) / 2 * sx

        # Quadrat in BILDkoordinaten -- dort ist die Anatomie unverzerrt
        crop_side = max(box_h, box_w) * (1 + 2 * args.margin)
        top, left = cy - crop_side / 2, cx - crop_side / 2
        log.append({
            "file": str(rel).replace("\\", "/"),
            "class": rel.parent.name,
            "split": rel.parts[0] if len(rel.parts) > 2 else "all",
            "img_h": img.shape[0], "img_w": img.shape[1],
            "crop_side": round(crop_side, 1),
            # >1 = das Bild wird vergroessert, <1 = verkleinert
            "zoom": args.size / max(1.0, crop_side),
            "mask_area_frac": float(mask_s.mean()),
            "mirrored": int(mirrored),
        })

        img_c = crop_pad(img.astype(np.float32) / 255.0, top, left,
                         crop_side, crop_side)
        img_c = sk_resize(img_c, (args.size, args.size), order=1,
                          anti_aliasing=True, preserve_range=True)

        # dasselbe Fenster, zurueckgerechnet ins Maskengitter: das Quadrat
        # aus Bildkoordinaten ist dort ein Rechteck
        mask_c = crop_pad(mask_s.astype(np.float32), top / sy, left / sx,
                          crop_side / sy, crop_side / sx)
        mask_c = sk_resize(mask_c, (args.size, args.size), order=1,
                           anti_aliasing=False, preserve_range=True)
        mask_c = (mask_c > 0.5).astype(np.float32)

        res = make_variants(img_c, mask_c)

        # Kontrollvariante ohne jede Segmentierung: fester zentraler Crop.
        # Schneidet sie im Vergleich gleich gut ab, traegt das U-Net nichts bei.
        if "centercrop" in variants:
            side = args.center_frac * min(img.shape)
            cc = crop_pad(img.astype(np.float32) / 255.0,
                          img.shape[0] / 2 - side / 2, img.shape[1] / 2 - side / 2,
                          side, side)
            cc = sk_resize(cc, (args.size, args.size), order=1,
                           anti_aliasing=True, preserve_range=True)
            res["centercrop"] = np.clip(cc * 255, 0, 255).astype(np.uint8)

        for v in variants:
            dst = args.out / v / rel.parent
            dst.mkdir(parents=True, exist_ok=True)
            Image.fromarray(res[v]).save(dst / (rel.stem + ".png"))

        if args.save_masks:
            dst = args.out / "mask_repaired" / rel.parent
            dst.mkdir(parents=True, exist_ok=True)
            Image.fromarray((mask_c * 255).astype(np.uint8)).save(
                dst / (rel.stem + ".png"))

        n_ok += 1
        n_mirrored += int(mirrored)

    print(f"fertig: {n_ok} verarbeitet, {n_skip} uebersprungen -> {args.out}")
    if args.mirror:
        print(f"  gespiegelt: {n_mirrored}")

    if not log:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "crop_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log[0].keys()))
        w.writeheader()
        w.writerows(log)
    print(f"  Protokoll: {csv_path}")

    # ---- Kontrolle: ist der Zoom klassenabhaengig? ----
    classes = sorted({r["class"] for r in log})
    if len(classes) == 2:
        pos = classes[-1]  # alphabetisch: NORMAL < PNEUMONIA
        y = np.array([1 if r["class"] == pos else 0 for r in log])
        print(f"\nZoom-Kontrolle (positiv = {pos}):")
        print(f"{'Groesse':<16}{classes[0]:>12}{classes[1]:>12}{'AUC':>8}")
        print("-" * 48)
        for key in ["crop_side", "zoom", "mask_area_frac"]:
            s = np.array([r[key] for r in log], float)
            print(f"{key:<16}{np.median(s[y == 0]):>12.3f}"
                  f"{np.median(s[y == 1]):>12.3f}{rank_auc(y, s):>8.3f}")
        print("\nZiel: AUC nahe 0.5. Bleibt sie hoch, ist der Crop selbst")
        print("klassenabhaengig und der Confounder nur verschoben, nicht weg.")


if __name__ == "__main__":
    main()
