"""
DICOM -> PNG. Einmalig, danach ist der Datensatz wie Kermany zu behandeln.

Warum ueberhaupt konvertieren: `pydicom.dcmread` plus JPEG-Dekodierung kostet
pro Bild ein Vielfaches eines PNG-Ladevorgangs, und das faellt in JEDER Epoche
erneut an. Bei 26 684 Bildern und der DirectML-Randbedingung `--workers 0`
(siehe READMEforMe, Windows spawnt sonst pro Worker einen frischen Torch-Import)
waere das der Flaschenhals -- die GPU wuerde auf den Decoder warten.

Bewusste Entscheidungen:

* **512x512, nicht 1024.** Trainiert wird auf 224. 512 laesst spaeter noch
  Raum fuer 384er-Versuche oder einen Lungen-Crop, ohne erneut 26 684 DICOMs
  anzufassen, und kostet ~7 GB statt ~25 GB. Bei 60 GB freiem Platz ist das
  die relevante Groesse.
* **Kein CLAHE, keine Normalisierung, kein Crop.** Das PNG ist eine getreue
  Kopie, nur kleiner. Jede Preprocessing-Entscheidung gehoert in den
  Dataset-Transform, wo sie pro Lauf umschaltbar und damit vergleichbar ist.
  Auf Kermany steckte CLAHE fest in der Konvertierung -- dadurch war nicht mehr
  trennbar, was Preprocessing und was Datensatz war.
* **Nur `stage_2_train_images`.** Der Testordner hat keine Labels und ist fuer
  uns wertlos.
* **Threads, keine Prozesse.** Die Arbeit ist ueberwiegend Datei-I/O und
  C-Dekodierung, beides gibt das GIL frei. Prozesse haetten unter Windows
  dasselbe spawn-Problem wie die DataLoader-Worker.

CLI:
  python rsna_prepare.py --dicom data/rsna/stage_2_train_images \
      --out data/rsna/png512 --size 512 --workers 8
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image


def preflight(files: list[Path]) -> None:
    """Ein Bild probeweise dekodieren, BEVOR 26 684 Dateien angefasst werden.

    RSNA-DICOMs sind JPEG-komprimiert. pydicom dekodiert das nur mit einem
    passenden Plugin; fehlt es, kommt der Fehler erst mitten im Lauf und sieht
    aus wie ein Datenproblem. Also lieber sofort und mit klarer Ansage.
    """
    import pydicom

    ds = pydicom.dcmread(str(files[0]))
    ts = ds.file_meta.TransferSyntaxUID
    print(f"Preflight an {files[0].name}")
    print(f"  TransferSyntax : {ts.name if hasattr(ts, 'name') else ts}")
    print(f"  Groesse        : {ds.Rows}x{ds.Columns}, "
          f"{ds.BitsStored} bit, {ds.PhotometricInterpretation}")
    try:
        arr = ds.pixel_array
    except Exception as e:                                   # noqa: BLE001
        print(f"\n  Dekodierung fehlgeschlagen: {type(e).__name__}: {e}")
        print("  Fehlendes JPEG-Plugin. Im aktiven venv installieren:")
        print("      pip install pylibjpeg pylibjpeg-libjpeg")
        print("  (nie waehrend eines laufenden Trainings -- s. READMEforMe/Hardware)")
        sys.exit(1)
    print(f"  Pixel          : dtype {arr.dtype}, "
          f"min {arr.min()}, max {arr.max()}\n")


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Auf 8 bit bringen -- ohne heimliche Kontrastanpassung.

    RSNA ist durchweg 8 bit MONOCHROME2, der erste Zweig greift also immer.
    Der zweite ist eine ehrliche Notbremse: er skaliert linear ueber den vollen
    Wertebereich. Das ist KEIN Windowing. Wuerde er je greifen, gehoert an
    dieser Stelle eine bewusste Fensterung hin, keine stille Autoskalierung --
    denn eine bildabhaengige Skalierung ist genau die Sorte globaler
    Kontrastnormierung, die auf Kermany zum Abkuerzungsweg wurde.
    """
    if arr.dtype == np.uint8:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, np.uint8)
    return (((arr.astype(np.float32) - lo) / (hi - lo)) * 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dicom", type=Path, default=Path("data/rsna/stage_2_train_images"))
    p.add_argument("--out", type=Path, default=Path("data/rsna/png512"))
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Probelauf. Achtung: die Dateireihenfolge ist NICHT "
                        "zufaellig (s. READMEforMe Phase 4).")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    files = sorted(args.dicom.glob("*.dcm"))
    if not files:
        raise SystemExit(f"Keine .dcm-Dateien unter {args.dicom}")
    if args.limit:
        files = files[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    preflight(files)

    todo = files if args.overwrite else [
        f for f in files if not (args.out / f"{f.stem}.png").exists()]
    print(f"{len(files)} DICOMs, {len(files) - len(todo)} bereits konvertiert, "
          f"{len(todo)} zu tun -> {args.out} ({args.size}x{args.size})")
    if not todo:
        return

    import pydicom

    done, failed = 0, []
    lock = Lock()
    t0 = time.time()

    def convert(f: Path) -> None:
        nonlocal done
        try:
            ds = pydicom.dcmread(str(f))
            img = Image.fromarray(to_uint8(ds.pixel_array)).convert("L")
            if img.size != (args.size, args.size):
                img = img.resize((args.size, args.size), Image.LANCZOS)
            # erst temporaer schreiben, dann umbenennen: ein Abbruch mitten im
            # Schreiben hinterlaesst sonst ein halbes PNG, das beim naechsten
            # Lauf als "schon fertig" gilt
            tmp = args.out / f"{f.stem}.png.tmp"
            img.save(tmp, "PNG", optimize=False)
            tmp.replace(args.out / f"{f.stem}.png")
        except Exception as e:                               # noqa: BLE001
            with lock:
                failed.append((f.name, f"{type(e).__name__}: {e}"))
        with lock:
            done += 1
            if done % 2000 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(todo)}  {el:5.0f}s  "
                      f"(noch ~{el / done * (len(todo) - done):.0f}s)")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(convert, todo))

    total_mb = sum(f.stat().st_size for f in args.out.glob("*.png")) / 1e6
    print(f"\nfertig in {time.time() - t0:.0f}s | {len(todo) - len(failed)} "
          f"geschrieben | {total_mb:.0f} MB in {args.out}")
    if failed:
        print(f"\n{len(failed)} FEHLGESCHLAGEN:")
        for name, err in failed[:10]:
            print(f"  {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
