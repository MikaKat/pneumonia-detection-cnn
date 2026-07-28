"""
Splits fuer RSNA -- geschichtet nach Label UND Projektion.

Drei Unterschiede zu `splits.py` (Kermany), jeder aus einem konkreten Fehler
der letzten Phase geboren:

1. **Geschichtet wird nach `label x ViewPosition`, nicht nur nach Label.**
   Der Confounder-Check hat gezeigt: `ViewPosition` allein trennt die Klassen
   mit AUC 0,706 und ist praktisch der GESAMTE Metadaten-Leak (innerhalb einer
   Projektion bleibt 0,553 uebrig, also Rauschen). Wenn die AP/PA-Quote zwischen
   den Folds schwankt, schwanken die Fold-AUCs aus einem Grund, der nichts mit
   dem Modell zu tun hat. Ausserdem braucht die spaetere Auswertung je Schicht
   in JEDEM Fold genug Faelle beider Klassen.

2. **Gespeichert werden `patientId`-Strings, keine Dateipfade.**
   Auf Kermany enthielt `splits.json` Windows-Backslashes; `parse_record` gab
   daraufhin still `None` zurueck statt zu krachen, und der innere Selektions-
   Split war eine Weile kaputt, ohne dass es auffiel. IDs sind
   plattformneutral -- den Pfad baut der Dataset-Loader aus Wurzel + ID + Endung.

3. **Ein echter Holdout wird abgetrennt.** Der offizielle `stage_2_test_images`-
   Ordner hat KEINE Labels (das war der Wettbewerbs-Testsatz), taugt also nicht
   als Ersatz. Also werden 15 % der gelabelten Bilder vor allem anderen
   weggeschlossen und bis zum Schluss nicht angefasst. Die 5 CV-Folds laufen auf
   den restlichen 85 %.

Gruppierung nach `patientId`: in RSNA ist das ein Bild je Patient, die
Gruppierung ist also faktisch wirkungslos. Sie bleibt trotzdem drin -- sie
kostet nichts und die Zusicherung "keine Gruppe in Train und Val" wird
tatsaechlich geprueft, statt angenommen.

CLI:
  python rsna_splits.py --csv data/rsna --headers qc/rsna/dicom_headers.csv \
      --out rsna_splits.json --folds 5 --holdout 0.15
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from rsna_data import load_labels, scan_headers


def build_table(csv_dir: Path, headers: Path | None, dicom_dir: Path | None,
                mode: str) -> pd.DataFrame:
    """Labels + ViewPosition in einer Tabelle."""
    lab = load_labels(csv_dir, mode=mode)
    if headers and Path(headers).exists():
        hdr = scan_headers(dicom_dir or Path("."), cache=headers)
    elif dicom_dir:
        hdr = scan_headers(dicom_dir, cache=headers)
    else:
        raise SystemExit("Weder --headers noch --dicom angegeben.")

    df = lab.merge(hdr[["patientId", "ViewPosition"]], on="patientId", how="inner")
    if len(df) < len(lab):
        print(f"WARNUNG: {len(lab) - len(df)} Bilder ohne Header, fallen raus.")
    df["ViewPosition"] = df["ViewPosition"].fillna("UNKNOWN").astype(str)
    # Der Schichtungsschluessel: Label und Projektion gemeinsam.
    df["stratum"] = df["label"].astype(str) + "|" + df["ViewPosition"]
    return df


def report(name: str, sub: pd.DataFrame, total: pd.DataFrame) -> None:
    """Vergleicht die Zusammensetzung einer Teilmenge mit der Gesamtmenge."""
    line = (f"  {name:<12}n={len(sub):>6}  pos={sub['label'].mean():.3f}"
            f"  (soll {total['label'].mean():.3f})")
    for vp in sorted(total["ViewPosition"].unique()):
        s = sub[sub["ViewPosition"] == vp]
        line += f"   {vp} {len(s) / max(len(sub), 1):.3f}/pos {s['label'].mean():.3f}"
    print(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--headers", type=Path, default=Path("qc/rsna/dicom_headers.csv"),
                   help="Header-Cache aus rsna_metadata_leak_check.py")
    p.add_argument("--dicom", type=Path, default=None,
                   help="nur noetig, wenn der Header-Cache fehlt")
    p.add_argument("--out", type=Path, default=Path("rsna_splits.json"))
    p.add_argument("--mode", default="clinical", choices=["clinical", "strict"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--holdout", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = build_table(args.csv, args.headers, args.dicom, args.mode)
    print(f"\n{len(df)} Bilder, Modus '{args.mode}', "
          f"{df['patientId'].nunique()} Patientengruppen")
    print("Schichten:", dict(sorted(Counter(df["stratum"]).items())))

    ids = df["patientId"].values
    y = df["label"].values
    strat = df["stratum"].values
    groups = df["patientId"].values          # = ids, s. Modul-Docstring

    # ---- 1. Holdout abtrennen ----------------------------------------------
    # StratifiedGroupKFold statt train_test_split: derselbe Mechanismus wie bei
    # den CV-Folds, also dieselben Garantien. n_splits ergibt den Anteil.
    n_out = max(2, round(1 / args.holdout))
    sgkf = StratifiedGroupKFold(n_splits=n_out, shuffle=True, random_state=args.seed)
    dev_idx, hold_idx = next(iter(sgkf.split(np.zeros(len(df)), strat, groups)))

    dev, hold = df.iloc[dev_idx], df.iloc[hold_idx]
    assert not (set(dev["patientId"]) & set(hold["patientId"]))
    print(f"\nHoldout (bis zum Schluss unangetastet, ~1/{n_out}):")
    report("holdout", hold, df)
    report("dev", dev, df)

    # ---- 2. CV-Folds auf der Entwicklungsmenge -----------------------------
    print(f"\n{args.folds} Folds auf der Entwicklungsmenge:")
    sgkf2 = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                 random_state=args.seed)
    folds, seen_val = [], set()
    for k, (tr, va) in enumerate(sgkf2.split(np.zeros(len(dev)),
                                             dev["stratum"].values,
                                             dev["patientId"].values)):
        tr_ids = dev.iloc[tr]["patientId"].tolist()
        va_ids = dev.iloc[va]["patientId"].tolist()
        assert not (set(tr_ids) & set(va_ids)), f"Gruppen-Leak in Fold {k}"
        assert not (set(va_ids) & set(hold["patientId"])), f"Holdout-Leak in Fold {k}"
        seen_val |= set(va_ids)
        folds.append({"train": tr_ids, "val": va_ids})
        report(f"fold {k} val", dev.iloc[va], df)

    assert seen_val == set(dev["patientId"]), "Val-Folds decken die dev-Menge nicht ab"

    # Kleinste Zelle: bestimmt, ob die Auswertung je Projektion ueberhaupt traegt.
    worst = min(
        len(dev.iloc[va][(dev.iloc[va]["ViewPosition"] == vp) &
                         (dev.iloc[va]["label"] == lb)])
        for _, va in sgkf2.split(np.zeros(len(dev)), dev["stratum"].values,
                                 dev["patientId"].values)
        for vp in df["ViewPosition"].unique() for lb in (0, 1))
    print(f"\nKleinste Zelle (Fold x Projektion x Klasse): {worst} Faelle")
    if worst < 50:
        print("  WARNUNG: zu wenig fuer eine belastbare AUC je Schicht.")

    payload = {
        "meta": {"dataset": "rsna", "mode": args.mode, "folds": args.folds,
                 "holdout_frac": args.holdout, "seed": args.seed,
                 "n_images": int(len(df)), "key": "patientId"},
        "labels": {str(i): int(v) for i, v in zip(ids, y)},
        "viewpos": dict(zip(df["patientId"], df["ViewPosition"])),
        "folds": folds,
        "holdout": hold["patientId"].tolist(),
    }
    args.out.write_text(json.dumps(payload))
    print(f"\ngespeichert: {args.out}")
    print("Schluessel sind patientId-Strings, keine Pfade -- der Loader baut")
    print("den Pfad als  <wurzel>/<patientId>.png  zusammen.")


if __name__ == "__main__":
    main()
