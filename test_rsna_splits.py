"""
Prueft eine fertige rsna_splits.json auf die Zusicherungen, die sie geben soll.

Die Assertions in `rsna_splits.py` laufen beim Erzeugen. Das hier laeuft auf der
GESCHRIEBENEN Datei -- also auf dem, was das Training tatsaechlich liest. Auf
Kermany lag genau dort der Fehler: die Datei enthielt Windows-Backslashes, der
Parser gab still `None` zurueck, und der innere Selektions-Split war wochenlang
kaputt, ohne dass eine einzige Assertion ansprang.

  python test_rsna_splits.py --splits rsna_splits.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

FAILED = []


def check(name: str, cond: bool, info: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FAILED.append(name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    p.add_argument("--images", type=Path, default=None,
                   help="optional: PNG-Ordner, prueft dass jede ID eine Datei hat")
    p.add_argument("--min-cell", type=int, default=50)
    args = p.parse_args()

    d = json.loads(args.splits.read_text())
    labels, viewpos, folds, hold = (d["labels"], d["viewpos"],
                                    d["folds"], set(d["holdout"]))
    all_ids = set(labels)
    print(f"\n{args.splits}: {len(all_ids)} Bilder, {len(folds)} Folds, "
          f"{len(hold)} im Holdout, Modus '{d['meta']['mode']}'")

    print("\nStruktur")
    check("Schluessel sind IDs, keine Pfade",
          not any(("/" in i or "\\" in i or i.endswith(".png")) for i in all_ids))
    check("viewpos deckt alle IDs ab", set(viewpos) == all_ids)
    check("Labels sind 0/1", set(labels.values()) <= {0, 1})

    print("\nDisjunktheit")
    dev_seen: Counter[str] = Counter()
    for k, f in enumerate(folds):
        tr, va = set(f["train"]), set(f["val"])
        check(f"Fold {k}: train n val leer", not (tr & va), f"{len(tr & va)}")
        check(f"Fold {k}: kein Holdout in train", not (tr & hold), f"{len(tr & hold)}")
        check(f"Fold {k}: kein Holdout in val", not (va & hold), f"{len(va & hold)}")
        check(f"Fold {k}: nur bekannte IDs", (tr | va) <= all_ids)
        dev_seen.update(va)

    dev = all_ids - hold
    check("Val-Folds decken die dev-Menge genau ab", set(dev_seen) == dev,
          f"fehlen {len(dev - set(dev_seen))}, zuviel {len(set(dev_seen) - dev)}")
    check("jede dev-ID genau einmal im Val", set(dev_seen.values()) == {1},
          str(dict(Counter(dev_seen.values()))))

    print("\nSchichtung (Sollwerte aus der Gesamtmenge)")
    pos_all = sum(labels.values()) / len(labels)
    ap_all = sum(v == "AP" for v in viewpos.values()) / len(viewpos)
    print(f"  gesamt: pos {pos_all:.3f}, AP-Anteil {ap_all:.3f}")
    for k, f in enumerate(folds):
        va = f["val"]
        pos = sum(labels[i] for i in va) / len(va)
        ap = sum(viewpos[i] == "AP" for i in va) / len(va)
        check(f"Fold {k} val: Positivrate nahe Soll", abs(pos - pos_all) < 0.02,
              f"{pos:.3f}")
        check(f"Fold {k} val: AP-Anteil nahe Soll", abs(ap - ap_all) < 0.02,
              f"{ap:.3f}")
    hpos = sum(labels[i] for i in hold) / len(hold)
    hap = sum(viewpos[i] == "AP" for i in hold) / len(hold)
    check("Holdout: Positivrate nahe Soll", abs(hpos - pos_all) < 0.02, f"{hpos:.3f}")
    check("Holdout: AP-Anteil nahe Soll", abs(hap - ap_all) < 0.02, f"{hap:.3f}")

    print("\nZellbesetzung (Fold x Projektion x Klasse)")
    worst, where = 10**9, ""
    for k, f in enumerate(folds):
        for vp in ("AP", "PA"):
            for lb in (0, 1):
                n = sum(1 for i in f["val"] if viewpos[i] == vp and labels[i] == lb)
                if n < worst:
                    worst, where = n, f"Fold {k} {vp} label={lb}"
    check(f"kleinste Zelle >= {args.min_cell}", worst >= args.min_cell,
          f"{worst} ({where})")

    if args.images:
        print("\nDateien")
        missing = [i for i in all_ids if not (args.images / f"{i}.png").exists()]
        check("jede ID hat ein PNG", not missing,
              f"{len(missing)} fehlen, z.B. {missing[:3]}")

    print("\n" + ("ALLE TESTS BESTANDEN" if not FAILED
                  else f"{len(FAILED)} FEHLGESCHLAGEN: {FAILED}"))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
