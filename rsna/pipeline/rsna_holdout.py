"""Der eine Blick auf den Holdout, Phase 10.

Was hier passiert
-----------------
Die 3812 Bilder, die am ersten Tag weggeschlossen wurden, werden von den fuenf
Fold-Modellen des Gewinnerarms gerechnet. Jede Ausgabe wird mit der Kurve
dieses Modells kalibriert, danach werden die fuenf WAHRSCHEINLICHKEITEN
gemittelt. Das ist das ausgelieferte Modell. Zusaetzlich wird das Kopffeld
gemittelt, ebenfalls nach der Sigmoid-Funktion.

Dieses Skript URTEILT NICHT. Es schreibt nur Vorhersagen auf die Platte. Das
Urteil steht in `rsna_phase10_auswertung.py` und richtet sich nach
`erklaerungen/29_phase10_final.md`, Abschnitt 9.

Warum das Skript sich selbst sperrt
-----------------------------------
Ein Holdout ist so lange etwas wert, wie er ungesehen ist. Wer ihn zweimal
rechnet, weil beim ersten Mal etwas nicht gefiel, hat ihn zum zweiten
Selektionssplit gemacht. Deshalb legt der Lauf neben die Ergebnisse eine
Sperrdatei mit Zeitstempel und den Pruefsummen der fuenf Gewichte. Ein zweiter
Lauf bricht ab. Wer ihn wirklich braucht (etwa weil der erste an fehlenden
Bildern gescheitert ist), setzt `--erneut`; dann steht in der Sperrdatei und in
jeder spaeteren Ausgabe, dass es nicht der erste Blick war.

Die Sperre ist keine Sicherheitsmassnahme gegen Boeswilligkeit. Sie ist eine
Bremse gegen die eigene Bequemlichkeit, und sie steht hier, weil genau diese
Bequemlichkeit der haeufigste Weg ist, auf dem ein Holdout still verbraucht
wird.

Drei Pruefungen vor der ersten Zahl
-----------------------------------
1. Die Kalibrierdatei muss existieren und zum Arm passen. Ohne sie gaebe es
   keine Wahrscheinlichkeiten, sondern nur Rohwerte, und die Schwelle waere
   bedeutungslos.
2. Das Siegel wird nachgeprueft: keine Holdout-Kennung darf in irgendeinem
   Trainings- oder Bewertungsteil eines Folds vorkommen. Das ist billig und
   faengt eine vertauschte Splitdatei.
3. Jedes Gewicht muss einen zweiten Kopf haben. Ein einkoepfiges Gewicht wuerde
   klaglos laden und stillschweigend ein anderes Modell ausliefern.

CLI:
  python rsna/pipeline/rsna_holdout.py --dml-index 1
  python rsna/pipeline/rsna_holdout.py --dml-index 1 --erneut   # nur im Notfall
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import _repo_path  # noqa: F401  (legt die Nachbarordner auf den Importpfad)

from rsna_train import (HEAD_GRID, RsnaDataset, build_transforms, make_model,
                        pick_device, predict)

ARM_TAG = "_p5head_ex"
FOLDS = [0, 1, 2, 3, 4]
SOLL_HOLDOUT_N = 3812
SOLL_SIZE = 224
SPERRE = ".holdout_verbraucht.json"


def abbruch(text: str) -> None:
    print(f"\nABBRUCH: {text}")
    sys.exit(1)


def pruefsumme(pfad: Path) -> str:
    h = hashlib.sha256()
    with pfad.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def logit(p, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def platt_apply(p, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * logit(p) + b)))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, default=Path("data/rsna/png512"))
    p.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    p.add_argument("--kalibrierung", type=Path,
                   default=Path("serving") / "model" / "kalibrierung_p10.json")
    p.add_argument("--out-dir", type=Path, default=Path("predictions_holdout"))
    p.add_argument("--size", type=int, default=SOLL_SIZE)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--dml-index", type=int, default=0)
    p.add_argument("--erneut", action="store_true",
                   help="die Sperre uebergehen. Der Lauf wird als NICHT der "
                        "erste Blick gekennzeichnet.")
    args = p.parse_args()

    print("=" * 78)
    print("DER EINE BLICK AUF DEN HOLDOUT")
    print("=" * 78)

    # ---- Sperre ----------------------------------------------------------
    sperre = args.out_dir / SPERRE
    if sperre.is_file() and not args.erneut:
        alt = json.loads(sperre.read_text(encoding="utf-8"))
        abbruch(f"der Holdout ist am {alt.get('wann')} bereits gerechnet worden.\n"
                f"         Ergebnisse: {args.out_dir}\n"
                f"         Ein zweiter Blick macht ihn zum Selektionssplit. Wer ihn "
                f"wirklich\n         braucht, setzt --erneut; der Lauf wird dann "
                f"als solcher gekennzeichnet.")
    erster_blick = not sperre.is_file()
    if not erster_blick:
        print("  ACHTUNG: --erneut gesetzt. Das ist NICHT der erste Blick.")

    # ---- Kalibrierung ----------------------------------------------------
    if not args.kalibrierung.is_file():
        abbruch(f"{args.kalibrierung} fehlt. Erst rsna/befunde/rsna_platt.py laufen "
                f"lassen; Kurve und Schwelle muessen VOR dem Holdout feststehen.")
    kal = json.loads(args.kalibrierung.read_text(encoding="utf-8"))
    if kal.get("arm") != ARM_TAG:
        abbruch(f"die Kalibrierdatei gehoert zu Arm {kal.get('arm')!r}, "
                f"gebraucht wird {ARM_TAG!r}")
    kurven = {int(e["fold"]): (float(e["a"]), float(e["b"])) for e in kal["platt"]}
    if sorted(kurven) != FOLDS:
        abbruch(f"die Kalibrierdatei kennt die Folds {sorted(kurven)}, "
                f"gebraucht werden {FOLDS}")
    print(f"  Kalibrierung aus {args.kalibrierung}")
    print(f"    Schwelle {kal['schwelle']:.4f}, {kal['schwelle_herkunft']}")

    # ---- Siegel ----------------------------------------------------------
    sp = json.loads(args.splits.read_text())
    labels = {k: int(v) for k, v in sp["labels"].items()}
    vpmap = sp["viewpos"]
    ids = list(sp["holdout"])
    if len(ids) != SOLL_HOLDOUT_N:
        abbruch(f"der Holdout hat {len(ids)} Bilder, erwartet {SOLL_HOLDOUT_N}")
    menge = set(ids)
    if len(menge) != len(ids):
        abbruch("der Holdout enthaelt eine Kennung doppelt")
    for k, fold in enumerate(sp["folds"]):
        for teil in ("train", "val"):
            ueber = menge & set(fold[teil])
            if ueber:
                abbruch(f"{len(ueber)} Holdout-Kennungen stehen in fold {k} "
                        f"'{teil}'. Das Siegel ist gebrochen.")
    print(f"  Siegel geprueft: {len(ids)} Kennungen, keine davon in einem der "
          f"fuenf Folds")

    fehlend = [i for i in ids if not (args.images / f"{i}.png").is_file()]
    if fehlend:
        abbruch(f"{len(fehlend)} Bilder fehlen unter {args.images}, "
                f"zuerst {fehlend[0]}")

    y = np.array([labels[i] for i in ids], dtype=float)
    vp = np.array([vpmap[i] for i in ids])
    print(f"  {len(ids)} Bilder, Praevalenz {y.mean():.4f}, "
          f"AP {int((vp == 'AP').sum())} / PA {int((vp == 'PA').sum())}")

    device, pin, dev_label = pick_device(args.device, args.dml_index)
    print(f"  Hardware: {dev_label}")
    print(f"  Bilder aus {args.images}, {args.size} Bildpunkte")

    # ---- rechnen ---------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ds = RsnaDataset(args.images, ids, labels, build_transforms(args.size, False))
    loader = DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                        pin_memory=pin)

    tabelle = {"patientId": ids, "y": y, "viewpos": vp}
    felder = {}
    summen = np.zeros(len(ids))
    summen_feld = np.zeros((len(ids), HEAD_GRID, HEAD_GRID))
    pruef = {}
    t_all = time.time()

    for k in FOLDS:
        ckpt = Path("checkpoints") / f"rsna_f{k}_s0_p5head_ex.pth"
        if not ckpt.is_file():
            abbruch(f"{ckpt} fehlt")
        pruef[ckpt.name] = pruefsumme(ckpt)

        model = make_model(device, head=True, grid=HEAD_GRID)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        fehlt, zuviel = model.load_state_dict(state, strict=False)
        if fehlt or zuviel:
            abbruch(f"{ckpt.name} passt nicht auf das zweikoepfige Modell.\n"
                    f"         fehlend {list(fehlt)[:4]}, ueberzaehlig "
                    f"{list(zuviel)[:4]}")

        t0 = time.time()
        pk, yk, fk = predict(model, loader, device, fields=True)
        if not np.allclose(yk, y):
            abbruch(f"Fold {k}: der Loader hat die Reihenfolge geaendert")
        if fk.shape[1:] != (HEAD_GRID, HEAD_GRID):
            abbruch(f"Fold {k}: das Kopffeld ist {fk.shape[1:]}, "
                    f"erwartet {(HEAD_GRID, HEAD_GRID)}")

        a, b = kurven[k]
        pc = platt_apply(pk, a, b)
        tabelle[f"p_roh_f{k}"] = pk
        tabelle[f"p_kal_f{k}"] = pc
        felder[f"feld_f{k}"] = fk.astype(np.float32)
        summen += pc
        summen_feld += fk
        print(f"  Fold {k}: {ckpt.name}  {time.time() - t0:.0f} s   "
              f"roh im Mittel {pk.mean():.4f} -> kalibriert {pc.mean():.4f}")
        del model

    tabelle["p_ens"] = summen / len(FOLDS)
    felder["feld_ens"] = (summen_feld / len(FOLDS)).astype(np.float32)

    df = pd.DataFrame(tabelle)
    ziel_csv = args.out_dir / "holdout.csv"
    df.to_csv(ziel_csv, index=False)
    np.savez_compressed(args.out_dir / "holdout_kopffelder.npz",
                        patientId=np.array(ids), **felder)

    nutzlast = {
        "wann": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "erster_blick": erster_blick,
        "arm": ARM_TAG,
        "n": len(ids),
        "images": str(args.images).replace("\\", "/"),
        "size": args.size,
        "device": dev_label,
        "checkpoints": pruef,
        "kalibrierung": str(args.kalibrierung).replace("\\", "/"),
        "schwelle": kal["schwelle"],
        "dauer_s": round(time.time() - t_all, 1),
    }
    sperre.write_text(json.dumps(nutzlast, indent=2), encoding="utf-8")

    print()
    print(f"  geschrieben: {ziel_csv}")
    print(f"               {args.out_dir / 'holdout_kopffelder.npz'}")
    print(f"               {sperre}   <- ab jetzt ist der Holdout verbraucht")
    print(f"  Gesamtdauer {time.time() - t_all:.0f} s")
    print()
    print("  Dieses Skript hat NICHT geurteilt. Das Urteil:")
    print("    venv\\Scripts\\python.exe rsna\\befunde\\rsna_phase10_auswertung.py")


if __name__ == "__main__":
    main()
