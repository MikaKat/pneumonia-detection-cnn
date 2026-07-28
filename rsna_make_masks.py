"""
Schritt 9a: U-Net-Lungenmasken fuer den RSNA-Datensatz erzeugen.

Warum ein eigenes Skript und nicht `segmentation/make_masks.py`?
  Jenes ist auf die Kermany-Ordnerstruktur `data/chest_xray/{split}/{klasse}/`
  festverdrahtet. RSNA liegt flach: `data/rsna/png512/{patientId}.png`, ohne
  Klassenordner (das Label steht in `stage_2_train_labels.csv`).

Die drei Regeln, die hier gelten:

  1. EIGENE VORVERARBEITUNG DES SEGMENTERS. Das U-Net hat Graustufen in [0,1]
     bei 256x256 gelernt -- ohne CLAHE, ohne Per-Bild-Standardisierung, ohne
     ImageNet-Normalisierung. Es bekommt genau das, NICHT die
     Klassifikator-Transform aus `rsna_train.build_transforms`.

  2. DOMAIN-SHIFT. Trainiert auf Montgomery/Shenzhen (Erwachsene,
     Tuberkulose-Screening), angewandt auf RSNA (Notaufnahme, viele
     Liegendaufnahmen). Deshalb QC-Vorschau UND Flaechenstatistik.

  3. TEILMENGE STATT ALLES. `--ids-from` nimmt die vorhandenen CAM-CSVs
     (~1500 Bilder statt 26 684).

ROH-CACHE -- der Grund fuer die zweite Fassung dieses Skripts
-------------------------------------------------------------
Der erste Lauf hat eine Lungenflaeche von 0,210 ergeben. Anatomisch sind in
einer frontalen Thoraxaufnahme eher 0,30-0,40 zu erwarten, und 28,5 % der
Bounding-Box-Flaeche lagen ausserhalb der Maske. Die Maske untersegmentiert
also -- dieselbe Richtung wie der bekannte Kermany-Fehler, nur schwaecher.

Um verschiedene Verfeinerungen (konvexe Huelle, Dilatation) zu vergleichen,
waere pro Variante ein neuer U-Net-Durchlauf noetig: 15 Minuten je Einstellung.
Das ist unnoetig, denn teuer ist nur der Forward-Pass; die Verfeinerung ist
Bildmorphologie auf einem fertigen Binaerbild. Deshalb wird die ROHE
U-Net-Ausgabe (256x256, binaer) einmal als gepackte Bits gecacht --
1500 Bilder ergeben rund 12 MB. Danach kostet jede weitere Variante Sekunden.

`--refine` waehlt die Verfeinerung, ohne die Modulglobalen in
`segmentation/mask_refine.py` zu veraendern. Das ist Absicht: eine Sweep-Schleife
soll mehrere Einstellungen in EINEM Prozess durchrechnen koennen, und globale
Schalter umzusetzen waere dabei eine Fehlerquelle, die still das falsche
Ergebnis liefert.

Ausgabe: `data/rsna/masks224/{patientId}.png`, 0/255, 224x224 -- genau das
Raster, in dem `pytorch_grad_cam` die Heatmap zurueckgibt. Maske und Heatmap
sind damit ohne Umrechnung deckungsgleich.

CLI:
  # erster Lauf: Masken + Roh-Cache anlegen (~15 min CPU)
  python rsna_make_masks.py --ids-from "predictions_rsna/cam_f*_s0.csv" \
      --raw-cache data/rsna/unet_raw256.npz

  # grosser Lauf: alle 22 872 Entwicklungsbilder, fortsetzbar
  #   --flush-every ist hier keine Feinheit, sondern Bedingung: ohne den
  #   Zwischenstand liegen 1,4 GiB Rohmasken bis zum Ende im Speicher, und
  #   ein Abbruch in Stunde drei kostet drei Stunden. Derselbe Aufruf noch
  #   einmal setzt den Lauf da fort, wo er stand.
  python rsna_make_masks.py --ids-from qc/dev_ids.csv \
      --masks data/rsna/masks224_dev --raw-cache data/rsna/unet_raw256.npz \
      --refine hull --dilate-px 8 --flush-every 2000 --device directml

  # Variante aus dem Cache, ohne U-Net (Sekunden)
  python rsna_make_masks.py --from-cache data/rsna/unet_raw256.npz \
      --refine hull --dilate-px 4 --masks data/rsna/masks224_hull4

  # nur Vorschau/Statistik neu
  python rsna_make_masks.py --ids-from "predictions_rsna/cam_f*_s0.csv" --qc-only
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SEG_SIZE = 256          # so hat das U-Net gelernt
OUT_SIZE = 224          # so kommt die Heatmap zurueck
DEFAULT_CKPT = Path("checkpoints/unet_best.pth")
REFINE_CHOICES = ("none", "default", "hull")


# --------------------------------------------------------------------------
# ID-Auswahl  (testbar ohne Torch)
# --------------------------------------------------------------------------

def ids_from_csvs(patterns: list[str]) -> list[str]:
    """patientIds aus einer oder mehreren CSVs sammeln, dedupliziert, sortiert."""
    paths: list[str] = []
    for pat in patterns:
        hits = sorted(globmod.glob(pat))
        if not hits:
            raise FileNotFoundError(f"Kein Treffer fuer Muster: {pat}")
        paths.extend(hits)

    ids: list[str] = []
    for p in paths:
        df = pd.read_csv(p)
        if "patientId" not in df.columns:
            raise ValueError(f"{p}: Spalte 'patientId' fehlt "
                             f"(vorhanden: {list(df.columns)})")
        ids.extend(df["patientId"].astype(str).tolist())
    return sorted(set(ids))


def ids_from_dir(root: Path) -> list[str]:
    return sorted(p.stem for p in Path(root).glob("*.png"))


def balanced_sample(splits_path: Path, n_per_class: int, seed: int = 0) -> list[str]:
    """n Bilder je Klasse aus der ENTWICKLUNGSMENGE ziehen (Holdout bleibt aussen vor).

    Gebraucht fuer jede Messung, die beide Klassen braucht -- etwa "verraten die
    Zuschnitt-Parameter das Label?". Der vorhandene Roh-Cache taugt dafuer
    nicht: er stammt aus den Grad-CAM-Stichproben, und Grad-CAM wurde nur auf
    POSITIVEN Bildern gerechnet. Alle 1500 Eintraege haben Target=1, eine AUC
    ist darauf nicht bestimmbar.

    Der Holdout wird ausdruecklich ausgeschlossen. Er ist fuer genau eine
    Auswertung reserviert; ihn fuer eine Vorbereitungsmessung anzufassen waere
    still verbrannte Evidenz.
    """
    sp = json.loads(Path(splits_path).read_text())
    holdout = set(sp.get("holdout", []))
    labels = {k: int(v) for k, v in sp["labels"].items() if k not in holdout}

    rng = np.random.default_rng(seed)
    out: list[str] = []
    for cls in (0, 1):
        pool = sorted(k for k, v in labels.items() if v == cls)
        take = min(n_per_class, len(pool))
        out.extend(rng.choice(pool, take, replace=False).tolist())
    return sorted(out)


def pending_jobs(ids: list[str], src: Path, dst: Path,
                 overwrite: bool,
                 cached: set[str] | None = None,
                 ) -> tuple[list[tuple[Path, Path]], int, int]:
    """(zu rechnende Paare, uebersprungen, fehlende Quellbilder).

    `cached` haelt die ids, die bereits IM ROH-CACHE stehen. Wird ein Cache
    gefuehrt, reicht die vorhandene Masken-PNG als Abbruchkriterium nicht: ein
    abgebrochener Lauf kann die PNG geschrieben, den Cache-Block danach aber
    nicht mehr geleert haben. Ohne diese Pruefung faellt genau dieses Bild bei
    der Fortsetzung durch beide Raster und fehlt im Cache dauerhaft -- still,
    und erst beim Zuschnitt bemerkbar.
    """
    jobs, skipped, missing = [], 0, 0
    for pid in ids:
        s = Path(src) / f"{pid}.png"
        d = Path(dst) / f"{pid}.png"
        if not s.exists():
            missing += 1
            continue
        done = d.exists() and (cached is None or pid in cached)
        if done and not overwrite:
            skipped += 1
            continue
        jobs.append((s, d))
    return jobs, skipped, missing


# --------------------------------------------------------------------------
# Roh-Cache: gepackte Bits  (testbar ohne Torch)
# --------------------------------------------------------------------------

def pack_masks(masks: np.ndarray) -> np.ndarray:
    """[n,256,256] bool -> [n, 8192] uint8. Faktor 8 gegenueber bool."""
    m = np.asarray(masks, dtype=bool).reshape(len(masks), -1)
    return np.packbits(m, axis=1)


def unpack_masks(packed: np.ndarray, shape=(SEG_SIZE, SEG_SIZE)) -> np.ndarray:
    n = len(packed)
    flat = np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=1)
    return flat[:, :shape[0] * shape[1]].reshape(n, *shape).astype(bool)


def save_raw_cache(path: Path, ids: list[str], masks: np.ndarray) -> None:
    """Neue Eintraege in einen vorhandenen Cache mischen (Lauf ist fortsetzbar).

    ATOMAR schreiben. Der Cache waechst auf ~190 MB; wird er an Ort und Stelle
    ueberschrieben und bricht der Lauf waehrend des Schreibens ab, ist nicht
    der letzte Block verloren, sondern alles. Deshalb erst daneben schreiben,
    dann umbenennen -- os.replace ist innerhalb eines Dateisystems atomar.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old_ids: list[str] = []
    old_packed = np.zeros((0, SEG_SIZE * SEG_SIZE // 8), np.uint8)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        old_ids = [str(s) for s in z["ids"]]
        old_packed = z["packed"]

    merged: dict[str, np.ndarray] = dict(zip(old_ids, old_packed))
    merged.update(dict(zip(ids, pack_masks(masks))))
    keys = sorted(merged)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, ids=np.array(keys),
                        packed=np.stack([merged[k] for k in keys]))
    os.replace(tmp, path)


def load_raw_cache(path: Path) -> tuple[list[str], np.ndarray]:
    z = np.load(Path(path), allow_pickle=False)
    return [str(s) for s in z["ids"]], z["packed"]


def cached_ids(path: Path | None) -> set[str]:
    """Welche ids stehen schon im Roh-Cache? Leere Menge, wenn es ihn nicht gibt."""
    if path is None or not Path(path).exists():
        return set()
    try:
        z = np.load(Path(path), allow_pickle=False)
        return {str(s) for s in z["ids"]}
    except Exception as e:                       # halb geschriebene Datei
        print(f"  WARNUNG: Roh-Cache nicht lesbar ({e}) -- wird neu aufgebaut.")
        return set()


# --------------------------------------------------------------------------
# Verfeinerung  (testbar ohne Torch)
# --------------------------------------------------------------------------

def refine_variant(raw: np.ndarray, mode: str = "default",
                   dilate_px: int = 0) -> np.ndarray:
    """Verfeinerung mit EXPLIZITEN Parametern statt Modulglobalen.

    mode:
      none     nur die rohe U-Net-Ausgabe (Kontrolle: was macht die
               Nachbearbeitung ueberhaupt?)
      default  saeubern + Symmetriefuellung (die bisherige Einstellung)
      hull     zusaetzlich konvexe Huelle je Lunge -- holt die Flaeche zurueck,
               die eine Konsolidierung dem Segmenter wegnimmt

    dilate_px weitet die fertige Maske elliptisch auf. Das ist bewusst grob:
    es geht nicht um anatomische Genauigkeit, sondern um die Frage, ob der
    gemessene Spielraum verschwindet, sobald die Maske nicht mehr zu klein ist.
    Wenn ja, war der Spielraum ein Maskenartefakt.
    """
    if mode not in REFINE_CHOICES:
        raise ValueError(f"mode muss in {REFINE_CHOICES} liegen, war {mode!r}")

    import cv2

    from segmentation.mask_refine import (_clean, _convex_hull_per_lung,
                                          _symmetry_fill)

    m = np.asarray(raw, dtype=bool)
    if mode != "none":
        m = _clean(m)
        if mode == "hull":
            m = _convex_hull_per_lung(m)
        m = _symmetry_fill(m)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m.astype(np.uint8), k).astype(bool)
    return m


def to_out(mask: np.ndarray, out_size: int = OUT_SIZE) -> np.ndarray:
    """Auf Klassifikatorraster bringen. NEAREST haelt die Maske binaer --
    bilinear erzeugte Grauwerte an den Raendern und verfaelschte die Flaeche."""
    import cv2
    m = (np.asarray(mask, dtype=bool).astype(np.uint8) * 255)
    return cv2.resize(m, (out_size, out_size), interpolation=cv2.INTER_NEAREST)


def area_report(areas: np.ndarray) -> dict:
    """Kennzahlen der Maskenflaeche -- die QC-Zahl neben dem Augenschein.

    Eine plausible Lunge belegt in einer frontalen Thoraxaufnahme grob 0,30-0,40
    der Bildflaeche. Der erste RSNA-Lauf lag bei 0,210: zu klein. Nahe 0 heisst,
    der Segmenter hat aufgegeben; ueber ~0,60, er hat das halbe Bild eingesammelt.
    """
    a = np.asarray(areas, dtype=float)
    if a.size == 0:
        return {}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "p05": float(np.percentile(a, 5)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "n_empty": int((a < 0.05).sum()),
        "n_huge": int((a > 0.60).sum()),
        "n_small": int((a < 0.22).sum()),      # unter anatomischer Erwartung
    }


# --------------------------------------------------------------------------
# Torch-Teil
# --------------------------------------------------------------------------

def _load_unet(ckpt: Path, device: str):
    """U-Net laden. Checkpoint IMMER auf die CPU, dann ins Modell kopieren.

    `map_location=<DirectML-Geraet>` stirbt mit
    "TypeError: '>=' not supported between instances of 'torch.device' and 'int'",
    weil Torch das device-Objekt an `torch_directml.device()` weiterreicht, die
    dort einen Integer-Index erwartet. Der Fehler sieht aus wie ein kaputter
    Checkpoint und ist keiner.

    `weights_only=True` unterdrueckt zugleich die FutureWarning; der Fallback
    faengt nur alte Torch-Versionen ohne dieses Argument ab.
    """
    import torch
    from segmentation.unet import UNet

    model = UNet(base_ch=32).to(device)
    try:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    except TypeError:                       # aeltere Torch-Versionen
        state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _preprocess(path: Path):
    """Exakt wie im Segmenter-Training: Graustufe, 256x256 bilinear, [0,1]."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    img = Image.open(path).convert("L")
    img = TF.resize(img, [SEG_SIZE, SEG_SIZE],
                    interpolation=InterpolationMode.BILINEAR)
    return TF.to_tensor(img)                     # [1,256,256] in [0,1]


def generate(jobs: list[tuple[Path, Path]], ckpt: Path, device: str, batch: int,
             mode: str, dilate_px: int,
             raw_cache: Path | None,
             flush_every: int = 0) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Masken rechnen und ablegen. Gibt (Flaechen, ids, rohe 256er-Masken) zurueck.

    `flush_every > 0` schreibt den Roh-Cache alle N Bilder weg und leert den
    Puffer. Zwei Gruende, beide bei 22 872 Bildern zwingend:

      Speicher. Ein Roh-Bild ist 256x256 bool = 64 KiB; alle zusammen 1,4 GiB,
      und `np.stack` am Ende verdoppelt das kurzzeitig.

      Verlust. Ohne Zwischenstand kostet ein Abbruch in Stunde drei die ganzen
      drei Stunden. Mit Zwischenstand kostet er hoechstens N Bilder, und
      `pending_jobs(..., cached=...)` nimmt den Lauf genau dort wieder auf.

    Bei flush_every=0 (Vorgabe, kleine Laeufe) verhaelt sich die Funktion
    unveraendert: der Puffer wird zurueckgegeben und main() speichert einmal.
    Bei flush_every>0 ist beim Verlassen bereits alles im Cache -- dann kommt
    ein LEERER Puffer zurueck, damit main() nicht ein zweites Mal schreibt.
    """
    import torch

    model = _load_unet(ckpt, device)
    areas, ids, raws = [], [], []
    n_flushed = 0
    total = len(jobs)
    t0 = time.time()

    def flush() -> int:
        """Puffer in den Cache mischen und leeren. Gibt die Anzahl zurueck."""
        nonlocal ids, raws
        if raw_cache is None or not ids:
            return 0
        n = len(ids)
        save_raw_cache(raw_cache, ids, np.stack(raws))
        ids, raws = [], []
        return n

    with torch.no_grad():
        for i in range(0, total, batch):
            chunk = jobs[i:i + batch]
            x = torch.stack([_preprocess(s) for s, _ in chunk]).to(device)
            pred = (torch.sigmoid(model(x)) > 0.5).cpu().numpy()[:, 0]
            for (src, out_path), p in zip(chunk, pred):
                out = to_out(refine_variant(p, mode, dilate_px))
                Image.fromarray(out, mode="L").save(out_path)
                areas.append(float((out > 127).mean()))
                if raw_cache is not None:
                    ids.append(src.stem)
                    raws.append(p)
            done = min(i + batch, total)
            if done % (batch * 10) == 0 or done == total:
                el = time.time() - t0
                rest = el / done * (total - done)
                print(f"    Masken {done}/{total}  "
                      f"({el / 60:.0f} min gelaufen, noch ~{rest / 60:.0f} min)",
                      flush=True)
            if flush_every and len(ids) >= flush_every:
                n_flushed += flush()
                print(f"    Zwischenstand gesichert ({n_flushed} in diesem Lauf)",
                      flush=True)

    if flush_every:
        n_flushed += flush()
        return np.asarray(areas), [], np.zeros((0, SEG_SIZE, SEG_SIZE), bool)

    raw_arr = np.stack(raws) if raws else np.zeros((0, SEG_SIZE, SEG_SIZE), bool)
    return np.asarray(areas), ids, raw_arr


def from_cache(cache: Path, dst: Path, mode: str, dilate_px: int,
               only: list[str] | None) -> np.ndarray:
    """Masken aus dem Roh-Cache neu verfeinern -- ohne Torch, in Sekunden."""
    ids, packed = load_raw_cache(cache)
    keep = set(only) if only else None
    Path(dst).mkdir(parents=True, exist_ok=True)

    areas = []
    for i, pid in enumerate(ids):
        if keep is not None and pid not in keep:
            continue
        raw = unpack_masks(packed[i:i + 1])[0]
        out = to_out(refine_variant(raw, mode, dilate_px))
        Image.fromarray(out, mode="L").save(Path(dst) / f"{pid}.png")
        areas.append(float((out > 127).mean()))
    return np.asarray(areas)


def measured_areas(ids: list[str], dst: Path) -> np.ndarray:
    out = []
    for pid in ids:
        p = Path(dst) / f"{pid}.png"
        if p.exists():
            out.append(float((np.array(Image.open(p)) > 127).mean()))
    return np.asarray(out)


# --------------------------------------------------------------------------
# QC
# --------------------------------------------------------------------------

def qc_preview(ids: list[str], src: Path, dst: Path, csv_dir: Path,
               out_png: Path, n_per_class: int = 4, seed: int = 0) -> None:
    """Vorschau mit ueberlagerter Maske, getrennt nach Label.

    Getrennt nach Label, weil genau dort der bekannte Fehler sitzt: auf Kermany
    wurden Pneumonie-Lungen UNTERsegmentiert (die Konsolidierung sieht dem
    Segmenter nicht nach Lunge aus). Wenn das wiederkommt, muss es sichtbar
    sein, bevor die Maske irgendetwas entscheidet.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = pd.read_csv(Path(csv_dir) / "stage_2_train_labels.csv")
    lab = (labels.groupby("patientId")["Target"].max()).to_dict()

    have = [i for i in ids if (Path(dst) / f"{i}.png").exists()]
    rng = np.random.default_rng(seed)
    rows: list[tuple[str, str]] = []
    for name, want in (("Target=0", 0), ("Target=1", 1)):
        pool = [i for i in have if lab.get(i) == want]
        if not pool:
            continue
        pick = rng.choice(pool, min(n_per_class, len(pool)), replace=False)
        rows.extend((name, str(p)) for p in pick)

    if not rows:
        print("  QC: keine Masken zum Anzeigen gefunden.")
        return

    fig, axes = plt.subplots(len(rows), 2, figsize=(6, 3 * len(rows)),
                             squeeze=False)
    for r, (name, pid) in enumerate(rows):
        img = np.array(Image.open(Path(src) / f"{pid}.png").convert("L")
                       .resize((OUT_SIZE, OUT_SIZE)))
        m = np.array(Image.open(Path(dst) / f"{pid}.png")) > 127
        axes[r][0].imshow(img, cmap="gray")
        axes[r][0].set_title(f"{name}  {pid[:8]}", fontsize=8)
        axes[r][1].imshow(img, cmap="gray")
        axes[r][1].imshow(m, cmap="Reds", alpha=0.35)
        axes[r][1].set_title(f"Maske  Flaeche {m.mean():.3f}", fontsize=8)
        for c in (0, 1):
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=90)
    plt.close(fig)
    print(f"  QC-Vorschau: {out_png}")


def area_leak_check(ids: list[str], dst: Path, csv_dir: Path) -> float | None:
    """Verraet die Maskenflaeche allein schon die Klasse?

    Genau der Leak, der auf Kermany gefunden wurde (lung_area AUC ~0,255).
    Muss VOR jedem Crop-Experiment bekannt sein: ein Crop auf eine Maske, deren
    Groesse die Klasse verraet, baut den Shortcut ins Bild ein.

    None, wenn nur eine Klasse vorliegt -- der CAM-Teilmenge fehlen die
    Negativen, weil Grad-CAM nur auf positiven Bildern gemessen wurde.
    """
    from sklearn.metrics import roc_auc_score

    labels = pd.read_csv(Path(csv_dir) / "stage_2_train_labels.csv")
    lab = (labels.groupby("patientId")["Target"].max()).to_dict()

    y, a = [], []
    for pid in ids:
        p = Path(dst) / f"{pid}.png"
        if pid in lab and p.exists():
            y.append(int(lab[pid]))
            a.append(float((np.array(Image.open(p)) > 127).mean()))
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, a))


def print_area(rep: dict) -> None:
    if not rep:
        return
    print(f"\nMaskenflaeche (Anteil des Bildes), n={rep['n']}:")
    print(f"  Mittel {rep['mean']:.3f} +- {rep['sd']:.3f} | P05 {rep['p05']:.3f} | "
          f"Median {rep['median']:.3f} | P95 {rep['p95']:.3f}")
    print(f"  leer (<0,05): {rep['n_empty']} | riesig (>0,60): {rep['n_huge']} | "
          f"unter anatomischer Erwartung (<0,22): {rep['n_small']}")
    if rep["mean"] < 0.26:
        print("  -> ZU KLEIN. Anatomisch sind ~0,30-0,40 zu erwarten. Eine zu kleine")
        print("     Maske erzeugt 'Maximum ausserhalb der Lunge' von selbst.")
        print("     Gegenmittel: --refine hull und/oder --dilate-px.")


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, default=Path("data/rsna/png512"))
    p.add_argument("--masks", type=Path, default=Path("data/rsna/masks224"))
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--ids-from", nargs="+", default=None,
                   help="CSV(s) mit Spalte patientId; Glob erlaubt")
    p.add_argument("--all", action="store_true",
                   help="alle PNGs unter --images (26 684 Stueck, ~1 h)")
    p.add_argument("--balanced-sample", type=int, default=None,
                   metavar="N",
                   help="N Bilder JE KLASSE aus der Entwicklungsmenge (Holdout "
                        "ausgeschlossen) -- fuer Messungen, die beide Klassen brauchen")
    p.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    p.add_argument("--refine", choices=REFINE_CHOICES, default="default",
                   help="none = roh | default = saeubern+Symmetrie | hull = zusaetzlich konvexe Huelle")
    p.add_argument("--dilate-px", type=int, default=0,
                   help="Maske um N Pixel (im 256er Raster) aufweiten")
    p.add_argument("--raw-cache", type=Path, default=None,
                   help="rohe U-Net-Ausgabe hierhin cachen (macht Varianten spaeter gratis)")
    p.add_argument("--from-cache", type=Path, default=None,
                   help="Masken aus dem Roh-Cache neu verfeinern, ohne U-Net")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--qc-only", action="store_true")
    p.add_argument("--qc-out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--flush-every", type=int, default=0, metavar="N",
                   help="Roh-Cache alle N Bilder sichern -- der Lauf wird damit "
                        "fortsetzbar. Fuer grosse Laeufe zwingend, sonst liegen "
                        "1,4 GiB Rohmasken bis zum Ende im Speicher.")
    args = p.parse_args(argv)

    if not (args.all or args.ids_from or args.from_cache or args.balanced_sample):
        p.error("--ids-from, --all, --balanced-sample oder --from-cache angeben")
    qc_out = args.qc_out or Path("qc") / f"rsna_mask_qc_{args.masks.name}.png"

    args.masks.mkdir(parents=True, exist_ok=True)

    # ---- Weg A: aus dem Roh-Cache, ohne Torch ------------------------------
    if args.from_cache:
        only = None
        if args.ids_from:
            only = ids_from_csvs(args.ids_from)
        elif args.balanced_sample:
            only = balanced_sample(args.splits, args.balanced_sample, args.seed)
        elif args.all:
            only = ids_from_dir(args.images)
        print(f"Aus Roh-Cache: {args.from_cache} | refine={args.refine} "
              f"dilate={args.dilate_px} -> {args.masks}")
        areas = from_cache(args.from_cache, args.masks, args.refine,
                           args.dilate_px, only)
        ids = ids_from_dir(args.masks)
        print(f"  {len(areas)} Masken geschrieben (kein U-Net noetig).")

    # ---- Weg B: U-Net rechnen ---------------------------------------------
    else:
        if args.balanced_sample:
            ids = balanced_sample(args.splits, args.balanced_sample, args.seed)
        elif args.all:
            ids = ids_from_dir(args.images)
        else:
            ids = ids_from_csvs(args.ids_from)
        print(f"Bilder in der Auswahl: {len(ids)}")
        have = cached_ids(args.raw_cache) if args.raw_cache is not None else None
        if have is not None:
            print(f"  bereits im Roh-Cache: {len(have)}")
        jobs, skipped, missing = pending_jobs(ids, args.images, args.masks,
                                              args.overwrite, cached=have)
        print(f"  zu rechnen: {len(jobs)} | vorhanden: {skipped} | "
              f"Quelle fehlt: {missing}")
        if missing:
            print("  ACHTUNG: fehlende Quellbilder deuten auf einen falschen "
                  "--images-Pfad oder eine unvollstaendige Konvertierung hin.")

        if args.qc_only:
            print("  --qc-only: es wird nichts gerechnet.")
        elif jobs:
            if not args.ckpt.exists():
                print(f"FEHLER: Checkpoint fehlt: {args.ckpt}")
                return 2
            print(f"  Geraet: {args.device} | Checkpoint: {args.ckpt} | "
                  f"refine={args.refine} dilate={args.dilate_px}")
            if args.flush_every:
                print(f"  Roh-Cache wird alle {args.flush_every} Bilder gesichert "
                      f"-- ein Abbruch kostet ab dann hoechstens so viele Bilder.")
            _, cids, raws = generate(jobs, args.ckpt, args.device, args.batch,
                                     args.refine, args.dilate_px, args.raw_cache,
                                     flush_every=args.flush_every)
            if args.raw_cache is not None and len(cids):
                save_raw_cache(args.raw_cache, cids, raws)
            if args.raw_cache is not None:
                n_now = len(cached_ids(args.raw_cache))
                print(f"  Roh-Cache: {args.raw_cache} ({n_now} Eintraege insgesamt)")
                print("  -> weitere Varianten jetzt mit --from-cache, ohne U-Net.")
        else:
            print("  nichts zu tun (alle Masken liegen vor).")

    # ---- QC: Zahlen zuerst, Bild danach -----------------------------------
    print_area(area_report(measured_areas(ids, args.masks)))

    auc = area_leak_check(ids, args.masks, args.csv)
    if auc is None:
        print("\nFlaechen-Leak: nicht bestimmbar (nur eine Klasse in der Auswahl).")
    else:
        print(f"\nFlaechen-Leak: AUC(Maskenflaeche -> Target) = {auc:.3f}")
        print("  0,5 = Flaeche verraet nichts. Auf Kermany war sie 0,255.")

    qc_preview(ids, args.images, args.masks, args.csv, qc_out, seed=args.seed)
    print("\nFertig. Erst die Vorschau ansehen, dann rsna_mask_sweep.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
