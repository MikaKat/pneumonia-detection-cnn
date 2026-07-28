"""
Schritt 9b: Diagnose ohne Retraining -- zeigt die Heatmap ueberhaupt in die Lunge?

Die Frage, die 11,5 Stunden Rechenzeit entscheidet
--------------------------------------------------
Gemessen ist: das Maximum der Grad-CAM liegt in 53,9 % der Faelle in einer
Bounding Box, Zufall waere 11,7 % -- Faktor 4,6. Die *Masse* der Karte liegt
aber nur zu 19,2 % in den Boxen. Die Karte zeigt grob richtig hin und ist
trotzdem diffus. Naheliegend waere ein Zuschnitt auf die Lunge (`crop`), damit
das Modell Bildrand, Schultern und Abdomen gar nicht erst sieht -- 11,5 h ueber
fuenf Folds. Vorher laesst sich mit den vorhandenen Checkpoints pruefen, ob er
ueberhaupt etwas bewirken KANN.

JEDE ZAHL BRAUCHT IHREN NENNER -- die Lehre aus dem ersten Lauf
----------------------------------------------------------------
Die erste Fassung dieses Skripts hat gemeldet: "80,7 % der Fehlschlaege zeigen
aus der Lunge heraus -> Crop gerechtfertigt". Diese Schwelle war ohne
Nullhypothese gesetzt, und das war falsch. Der Zufallswert fuer "ausserhalb"
ist 1 - Lungenflaeche, hier 0,792. Der Vorsprung betrug damit +0,014 +- 0,100
(t = 0,32 ueber fuenf Folds) -- also nichts. Zum Vergleich: bei den TREFFERN
lag das Maximum zu 0,858 in der Lunge gegen einen Zufall von 0,212.

Das Modell ist zweigipflig: entweder es findet die Pathologie (Maximum in der
Lunge und in der Box), oder sein Maximum ist bezueglich der Anatomie Rauschen.
Ein systematisches "schaut auf Rippen, Zwerchfell, Bildrand" existiert nicht --
passend zur Ecken-Ablation von -0,0001.

Deshalb steht ab jetzt neben JEDER Aussenzahl ihre Baseline, und das Verdikt
haengt am Vorsprung, nicht am Rohwert. Das gilt auch fuer den Spielraum: eine
reine Zufalls-Heatmap gewinnt durch die Beschraenkung auf die Lunge +0,264,
weil ihr Platz weggenommen wird. Ein beobachteter Gewinn muss dagegen gehalten
werden, nicht gegen Null.

ZWEI KONTROLLEN VOR JEDEM BEFUND
---------------------------------
1. `box_in_lung` -- welcher Anteil der Bounding Box liegt in der Maske? Auf
   Kermany wurden Pneumonie-Lungen untersegmentiert, weil eine Konsolidierung
   dem U-Net nicht nach Lunge aussieht. Dann faellt die Pathologie aus der
   Maske und "Maximum ausserhalb der Lunge" entsteht von selbst -- zirkulaer.
2. `lung_area` -- anatomisch sind 0,30-0,40 zu erwarten. Der erste Lauf ergab
   0,210. Eine zu kleine Maske treibt jede Aussenzahl UND den Spielraum nach
   oben, ohne dass das Modell etwas damit zu tun haette.

HEAT-CACHE
----------
Die Heatmaps werden als float16 gecacht (~12 MB je Fold). Damit ist der
Vergleich verschiedener Maskenvarianten ein Tabellen-Lookup in Sekunden statt
20 Minuten CAM-Rechnung je Variante -- siehe `rsna_mask_sweep.py`. Dass das in
der ersten Fassung fehlte, war ein Konstruktionsfehler.

CLI:
  python rsna_cam_lung_check.py --folds 0 1 2 3 4 --n 120 --cache-heat
  python rsna_cam_lung_check.py --folds 0 --n 300 --masks data/rsna/masks224_hull4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

LUNG_AREA_MIN = 0.26        # darunter ist die Maske anatomisch unplausibel klein
BOX_IN_LUNG_MIN = 0.60      # darunter schneidet die Maske die Pathologie weg


# --------------------------------------------------------------------------
# Geometrie und Auswertung  (torch-frei, damit testbar und im Sweep nutzbar)
# --------------------------------------------------------------------------

def box_mask(boxes, size: int, box_space: int) -> np.ndarray:
    """Bounding Boxes aus dem 1024er-DICOM-Raster ins Modellraster (size)."""
    s = size / box_space
    m = np.zeros((size, size), bool)
    for bx, by, bw, bh in boxes:
        y0, y1 = max(int(by * s), 0), int((by + bh) * s)
        x0, x1 = max(int(bx * s), 0), int((bx + bw) * s)
        m[y0:y1, x0:x1] = True
    return m


def analyse_one(heat: np.ndarray, box: np.ndarray, lung: np.ndarray) -> dict | None:
    """Alle Kennzahlen fuer ein Bild.

    `peak_in_box_lungrestricted` ist der Optimismus-Deckel: das Maximum wird nur
    noch INNERHALB der Lunge gesucht. Das ist NICHT, was ein auf Zuschnitten
    trainiertes Modell tut -- ein solches lernt andere Gewichte. Es ist die
    Obergrenze dessen, was allein das Wegnehmen des Sichtfelds braechte.

    `null_restricted` ist der Nenner dazu: die Trefferquote eines ZUFAELLIGEN
    Punktes innerhalb der Lunge, also (Box geschnitten Lunge) / Lunge. Ohne
    diese Zahl sieht jeder Gewinn nach Modellverdienst aus, obwohl er blosse
    Flaechenrechnung sein kann.

    Die Peak-Koordinaten werden mitgeschrieben, damit eine andere Maske
    nachtraeglich ohne neue CAM-Rechnung bewertet werden kann.
    """
    heat = np.clip(np.asarray(heat, dtype=float), 0, None)
    total = heat.sum()
    if total <= 0:
        return None

    yx = np.unravel_index(int(np.argmax(heat)), heat.shape)

    if lung.any():
        masked = np.where(lung, heat, -1.0)
        yx_l = np.unravel_index(int(np.argmax(masked)), masked.shape)
        peak_lr = bool(box[yx_l])
    else:
        yx_l = yx
        peak_lr = bool(box[yx])           # keine Maske -> keine Einschraenkung

    box_a = float(box.mean())
    lung_a = float(lung.mean())
    inter = float((box & lung).mean())
    return {
        "peak_y": int(yx[0]), "peak_x": int(yx[1]),
        "peak_lung_y": int(yx_l[0]), "peak_lung_x": int(yx_l[1]),
        "peak_in_box": bool(box[yx]),
        "peak_in_lung": bool(lung[yx]),
        "peak_in_box_lungrestricted": peak_lr,
        "mass_in_box": float(heat[box].sum() / total),
        "mass_in_lung": float(heat[lung].sum() / total),
        "box_area": box_a,
        "lung_area": lung_a,
        "box_in_lung": float((box & lung).sum() / box.sum()) if box.any() else np.nan,
        # Nenner fuer den Spielraum: Zufallstreffer frei bzw. auf die Lunge beschraenkt
        "null_free": box_a,
        "null_restricted": float(min(inter / lung_a, 1.0)) if lung_a > 0 else box_a,
    }


def summarise(df: pd.DataFrame) -> dict:
    """Fold-Zusammenfassung. Jede Quote steht neben ihrer Baseline."""
    d = df.dropna(subset=["peak_in_box"])
    miss = d[~d["peak_in_box"].astype(bool)]
    hit = d[d["peak_in_box"].astype(bool)]

    out = {
        "n": int(len(d)),
        "box_in_lung": float(d["box_in_lung"].mean()),
        "lung_area": float(d["lung_area"].mean()),
        "box_area": float(d["box_area"].mean()),
        "peak_in_box": float(d["peak_in_box"].mean()),
        "peak_in_lung": float(d["peak_in_lung"].mean()),
        "peak_in_lung_lift": float(d["peak_in_lung"].mean() - d["lung_area"].mean()),
        "mass_in_lung": float(d["mass_in_lung"].mean()),
        "mass_in_box": float(d["mass_in_box"].mean()),
        "n_miss": int(len(miss)),
        "hit_in_lung": float(hit["peak_in_lung"].mean()) if len(hit) else np.nan,
        "hit_in_lung_null": float(hit["lung_area"].mean()) if len(hit) else np.nan,
    }

    # Der Bruch, um den es geht -- Rohwert UND Nenner.
    if len(miss):
        out["miss_outside_lung"] = float((~miss["peak_in_lung"].astype(bool)).mean())
        out["miss_outside_null"] = float(1 - miss["lung_area"].mean())
    else:
        out["miss_outside_lung"] = np.nan
        out["miss_outside_null"] = np.nan
    out["miss_outside_lift"] = out["miss_outside_lung"] - out["miss_outside_null"]

    out["peak_in_box_lungrestricted"] = float(d["peak_in_box_lungrestricted"].mean())
    out["crop_headroom"] = out["peak_in_box_lungrestricted"] - out["peak_in_box"]
    # Was dieselbe Beschraenkung einer Zufalls-Heatmap braechte:
    out["null_free"] = float(d["null_free"].mean())
    out["null_restricted"] = float(d["null_restricted"].mean())
    out["headroom_null"] = out["null_restricted"] - out["null_free"]

    # Dieselbe Groesse, anders gelesen -- und diese Lesart ist die belastbare.
    #
    # Zugewinne zu vergleichen (Modell +0,080 gegen Zufall +0,264) ist
    # angreifbar: das Modell startet bei 0,530, der Zufall bei 0,117, hat also
    # weniger Luft nach oben. Der Einwand "Deckeneffekt" waere berechtigt.
    #
    # Deshalb stattdessen zwei VORSPRUENGE, jeder gegen seine eigene Baseline:
    #   frei         Trefferquote - Boxflaeche
    #   beschraenkt  Trefferquote(nur Lunge) - (Box UND Lunge)/Lunge
    # Beides sind Abstaende zur passenden Null und damit direkt vergleichbar.
    # Algebraisch ist die Differenz identisch mit headroom_vs_null -- aber in
    # dieser Form ist sie gegen den Deckeneinwand immun.
    out["lift_free"] = out["peak_in_box"] - out["null_free"]
    out["lift_restricted"] = out["peak_in_box_lungrestricted"] - out["null_restricted"]
    out["lift_delta"] = out["lift_restricted"] - out["lift_free"]
    out["headroom_vs_null"] = out["lift_delta"]
    return out


def cv_mean(rows: list[dict], key: str) -> tuple[float, float]:
    """Mittel +- SD ueber Folds. Nach dem Fold-0-gegen-Fold-1-Befund ist eine
    Einzelfoldzahl nichts wert -- Absolutwerte nur als CV-Mittel."""
    v = np.array([r[key] for r in rows if key in r and not np.isnan(r[key])],
                 dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def paired_t(rows: list[dict], key: str) -> float:
    """Gepaartes t einer Differenzspalte ueber die Folds. df = k-1, bei fuenf
    Folds ist |t| > 2,78 die 5-%-Grenze. Bewusst ohne scipy: eine einzelne
    t-Statistik rechtfertigt keine zusaetzliche Abhaengigkeit."""
    v = np.array([r[key] for r in rows if key in r and not np.isnan(r[key])],
                 dtype=float)
    if v.size < 2:
        return float("nan")
    sd = v.std(ddof=1)
    if sd == 0:
        return float("inf") if v.mean() != 0 else 0.0
    return float(v.mean() / (sd / np.sqrt(v.size)))


# --------------------------------------------------------------------------
# Torch-Teil
# --------------------------------------------------------------------------

def load_lung(masks: Path, pid: str, size: int) -> np.ndarray | None:
    p = Path(masks) / f"{pid}.png"
    if not p.exists():
        return None
    m = np.array(Image.open(p).convert("L"))
    if m.shape != (size, size):
        # Kein stilles Resize: die Maske wird bewusst in Modellaufloesung
        # erzeugt. Passt sie nicht, stimmt eine Annahme nicht.
        raise ValueError(f"{p}: Maske ist {m.shape}, erwartet ({size}, {size}). "
                         f"rsna_make_masks.py mit passender OUT_SIZE laufen lassen.")
    return m > 127


def run_fold(fold: int, args) -> pd.DataFrame:
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

    from rsna_train import BOX_SPACE, build_transforms, load_boxes, make_model

    ckpt = Path(f"checkpoints/rsna_f{fold}_s{args.seed}.pth")
    cam_csv = Path(args.pred_dir) / f"cam_f{fold}_s{args.seed}.csv"
    if not ckpt.exists():
        print(f"  Fold {fold}: Checkpoint fehlt ({ckpt}) -- uebersprungen.")
        return pd.DataFrame()
    if not cam_csv.exists():
        print(f"  Fold {fold}: CAM-CSV fehlt ({cam_csv}) -- uebersprungen.")
        return pd.DataFrame()

    # Dieselben Bilder wie in der berichteten Zahl, nicht neu gewuerfelt.
    stored = pd.read_csv(cam_csv)
    ids = stored["patientId"].astype(str).tolist()
    if args.n and args.n < len(ids):
        rng = np.random.default_rng(args.seed)
        ids = list(rng.choice(ids, args.n, replace=False))

    boxes = load_boxes(args.csv)
    model = make_model(torch.device("cpu"))
    try:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    tf = build_transforms(args.size, False)

    rows, no_mask = [], 0
    heat_ids, heats = [], []
    for j, pid in enumerate(ids, 1):
        lung = load_lung(args.masks, pid, args.size)
        if lung is None:
            no_mask += 1
            continue
        img = Image.open(Path(args.images) / f"{pid}.png").convert("L")
        x = tf(img).unsqueeze(0)
        heat = np.clip(cam(input_tensor=x,
                           targets=[BinaryClassifierOutputTarget(1)])[0], 0, None)
        r = analyse_one(heat, box_mask(boxes.get(pid, []), args.size, BOX_SPACE),
                        lung)
        if r is None:
            continue
        r["patientId"] = pid
        rows.append(r)
        if args.cache_heat:
            heat_ids.append(pid)
            heats.append(heat.astype(np.float16))
        if j % 50 == 0:
            print(f"    Fold {fold}: {j}/{len(ids)}")

    if no_mask:
        print(f"  Fold {fold}: {no_mask} Bilder ohne Maske uebersprungen "
              f"-- rsna_make_masks.py fuer diese IDs nachziehen.")

    if args.cache_heat and heats:
        out = Path(args.pred_dir) / f"cam_heat_f{fold}_s{args.seed}.npz"
        np.savez_compressed(out, ids=np.array(heat_ids), heat=np.stack(heats))
        print(f"  Fold {fold}: Heat-Cache {out.name} "
              f"({len(heats)} Karten, {out.stat().st_size / 1e6:.0f} MB)")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["fold"] = fold

    # Gegenprobe gegen die gespeicherte Zahl: dieselben Bilder, derselbe
    # Checkpoint, dieselbe Transform -- die Trefferquote MUSS wieder herauskommen.
    merged = df.merge(stored[["patientId", "hit"]], on="patientId", how="inner")
    if len(merged):
        agree = float((merged["peak_in_box"] == merged["hit"]).mean())
        print(f"  Fold {fold}: Reproduktion der gespeicherten Trefferquote "
              f"{agree:.3f} ueber {len(merged)} Bilder"
              + ("" if agree > 0.98 else "   <-- ACHTUNG, sollte ~1.000 sein"))
    return df


# --------------------------------------------------------------------------
# Bericht
# --------------------------------------------------------------------------

def _line(label: str, m: float, s: float, signed: bool = False) -> str:
    fmt = "%+.3f" if signed else "%.3f"
    return f"  {label:<46} " + (fmt % m) + f" +- {s:.3f}"


def report(per_fold: list[dict]) -> None:
    if not per_fold:
        print("Keine Ergebnisse.")
        return

    print("\n" + "=" * 76)
    print("KONTROLLEN ZUERST: taugt die Maske als Massstab?")
    print("=" * 76)
    bil, bil_s = cv_mean(per_fold, "box_in_lung")
    la, la_s = cv_mean(per_fold, "lung_area")
    print(_line("Bounding Box innerhalb der Maske", bil, bil_s))
    print(_line("Lungenflaeche (anatomisch ~0,30-0,40)", la, la_s))

    mask_ok = True
    if bil < BOX_IN_LUNG_MIN:
        mask_ok = False
        print(f"  -> box_in_lung < {BOX_IN_LUNG_MIN}: die Maske schneidet die "
              f"Pathologie weg.")
    if la < LUNG_AREA_MIN:
        mask_ok = False
        print(f"  -> Lungenflaeche < {LUNG_AREA_MIN}: die Maske ist zu klein. Sie")
        print("     erzeugt 'Maximum ausserhalb der Lunge' und Spielraum von selbst.")
        print("     Gegenmittel: rsna_make_masks.py --refine hull --dilate-px N")
    if mask_ok:
        print("  -> beide Kontrollen bestanden.")

    print("\n" + "=" * 76)
    print("BEFUND -- jede Quote neben ihrer Baseline")
    print("=" * 76)
    for k, kn, label in [
        ("peak_in_box", "box_area", "Maximum in einer Box"),
        ("peak_in_lung", "lung_area", "Maximum in der Lunge"),
        ("hit_in_lung", "hit_in_lung_null", "  davon: Treffer, Max in der Lunge"),
        ("mass_in_box", "box_area", "Heatmap-Masse in den Boxen"),
        ("mass_in_lung", "lung_area", "Heatmap-Masse in der Lunge"),
    ]:
        m, s = cv_mean(per_fold, k)
        n, _ = cv_mean(per_fold, kn)
        print(f"  {label:<40} {m:.3f} +- {s:.3f}   Zufall {n:.3f}   "
              f"Vorsprung {m - n:+.3f}")

    print("\n" + "-" * 76)
    print("DIE ENTSCHEIDENDE FRAGE: schaut das Modell bei Fehlschlaegen AUS der Lunge?")
    print("-" * 76)
    mo, mo_s = cv_mean(per_fold, "miss_outside_lung")
    mn, _ = cv_mean(per_fold, "miss_outside_null")
    ml, ml_s = cv_mean(per_fold, "miss_outside_lift")
    t_miss = paired_t(per_fold, "miss_outside_lift")
    print(f"  Fehlschlag, Maximum ausserhalb der Lunge   {mo:.3f} +- {mo_s:.3f}")
    print(f"  Zufall dafuer (1 - Lungenflaeche)          {mn:.3f}")
    print(f"  VORSPRUNG                                  {ml:+.3f} +- {ml_s:.3f}"
          f"   (gepaartes t = {t_miss:+.2f}, |t|>2,78 = p<0,05)")

    print("\n" + "-" * 76)
    print("SPIELRAUM: was braechte allein das Wegnehmen des Sichtfelds?")
    print("-" * 76)
    hm, hm_s = cv_mean(per_fold, "peak_in_box_lungrestricted")
    bm, _ = cv_mean(per_fold, "peak_in_box")
    cm, cm_s = cv_mean(per_fold, "crop_headroom")
    nm, _ = cv_mean(per_fold, "headroom_null")
    vm, vm_s = cv_mean(per_fold, "headroom_vs_null")
    print(f"  Treffer, Maximum nur in der Lunge gesucht   {hm:.3f} +- {hm_s:.3f}")
    print(f"  gegenueber jetzt                            {bm:.3f}")
    print(f"  beobachteter Zugewinn                       {cm:+.3f} +- {cm_s:.3f}"
          f"   (t = {paired_t(per_fold, 'crop_headroom'):+.2f})")
    print(f"  Zugewinn einer ZUFALLS-Heatmap              {nm:+.3f}"
          "   <- reine Flaechenrechnung")

    # Zugewinne zu vergleichen laedt den Einwand "Deckeneffekt" ein: das Modell
    # startet hoeher, hat also weniger Luft. Zwei Vorspruenge gegen die je
    # eigene Baseline haben dieses Problem nicht.
    lf, lf_s = cv_mean(per_fold, "lift_free")
    lr, lr_s = cv_mean(per_fold, "lift_restricted")
    print("\n  Deckeneffekt-fest formuliert -- zwei Vorspruenge, je gegen die"
          " eigene Null:")
    print(f"    frei         {bm:.3f} - {cv_mean(per_fold, 'null_free')[0]:.3f}"
          f" = {lf:+.3f} +- {lf_s:.3f}")
    print(f"    beschraenkt  {hm:.3f} - "
          f"{cv_mean(per_fold, 'null_restricted')[0]:.3f} = {lr:+.3f} +- {lr_s:.3f}")
    print(f"    DIFFERENZ    {vm:+.3f} +- {vm_s:.3f}   "
          f"(t = {paired_t(per_fold, 'lift_delta'):+.2f})")
    print("  Negativ heisst: die Beschraenkung auf die Lunge macht das Maximum")
    print("  WENIGER informativ, nicht mehr. (Obergrenze -- ein auf Zuschnitten")
    print("  trainiertes Modell lernt andere Gewichte.)")

    print("\n" + "=" * 76)
    print("LESART")
    print("=" * 76)
    if not mask_ok:
        # Reihenfolge zaehlt: eine zu kleine oder verschobene Maske erzeugt den
        # Aussen-Befund von selbst. Die Lesart waere zirkulaer -- also keine.
        print("  KEINE LESART, solange die Maskenkontrollen nicht bestanden sind.")
        print("  Der Aussen-Anteil und der Spielraum sind dann Artefakte der")
        print("  Segmentierung, kein Befund ueber das Modell. Erst die Maske")
        print("  reparieren (rsna_mask_sweep.py vergleicht Varianten in Sekunden),")
        print("  dann wieder hier her.")
    elif np.isnan(ml):
        print("  Keine Fehlschlaege in der Stichprobe -- nichts zu entscheiden.")
    elif abs(t_miss) < 2.78:
        print(f"  Der Aussen-Anteil ({mo:.3f}) ist von seinem Zufallswert ({mn:.3f})")
        print(f"  nicht zu unterscheiden (Vorsprung {ml:+.3f}, t = {t_miss:+.2f}).")
        print("  Das Modell schaut bei Fehlschlaegen nicht systematisch aus der")
        print("  Lunge heraus -- sein Maximum ist dann schlicht uninformativ.")
        print("  -> Der Mechanismus, auf dem die Crop-Hoffnung beruht, ist nicht da.")
        print("  -> Schritt 3 nicht auf dieser Grundlage starten.")
    elif ml > 0 and vm > 0.02:
        print("  Der Aussen-Anteil liegt ueber seinem Zufallswert UND der Spielraum")
        print("  uebertrifft den rein geometrischen Zugewinn. Beides zusammen ist")
        print("  der Mechanismus, den ein Crop adressiert.")
        print("  -> Schritt 3 (crop, 5 Folds GEPAART je Fold) ist gerechtfertigt.")
    else:
        print("  Gemischt: der Aussen-Anteil weicht vom Zufall ab, der Spielraum")
        print("  bleibt aber im Rahmen der reinen Flaechenrechnung. Wenn ueberhaupt,")
        print("  dann gepaart je Fold messen -- ein Mittelwertvergleich koennte einen")
        print("  Effekt dieser Groesse nicht von der Fold-Streuung trennen.")
    print("=" * 76)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, default=Path("data/rsna/png512"))
    p.add_argument("--masks", type=Path, default=Path("data/rsna/masks224"))
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--pred-dir", type=Path, default=Path("predictions_rsna"))
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--n", type=int, default=120,
                   help="Bilder je Fold (0 = alle aus der CAM-CSV)")
    p.add_argument("--cache-heat", action="store_true",
                   help="Heatmaps als float16 cachen -- macht Maskenvarianten gratis")
    p.add_argument("--out", type=Path,
                   default=Path("predictions_rsna/cam_lung.csv"))
    args = p.parse_args(argv)

    if not Path(args.masks).exists():
        print(f"FEHLER: Maskenordner fehlt: {args.masks}")
        print("  Zuerst:  python rsna_make_masks.py "
              "--ids-from \"predictions_rsna/cam_f*_s0.csv\" "
              "--raw-cache data/rsna/unet_raw256.npz")
        return 2

    frames, per_fold = [], []
    for f in args.folds:
        print(f"\nFold {f} (CPU, ein paar Sekunden je Bild)...")
        df = run_fold(f, args)
        if df.empty:
            continue
        frames.append(df)
        s = summarise(df)
        s["fold"] = f
        per_fold.append(s)
        print(f"  in Box {s['peak_in_box']:.3f} | in Lunge {s['peak_in_lung']:.3f} "
              f"(Zufall {s['lung_area']:.3f}) | Fehlschlag aussen "
              f"{s['miss_outside_lung']:.3f} (Zufall {s['miss_outside_null']:.3f})")

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(args.out, index=False)
        pd.DataFrame(per_fold).to_csv(
            args.out.with_name(args.out.stem + "_byfold.csv"), index=False)
        print(f"\nRohdaten: {args.out}")

    report(per_fold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
