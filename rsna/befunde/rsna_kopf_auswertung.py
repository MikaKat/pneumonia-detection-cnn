"""
Phase 5: das Kopffeld gegen den Lagepriore. Der Rauchtest vor jedem Vergleich.

WOZU
----
Die Roadmap verlangt vor jedem Vergleich einen Test auf die WIRKUNG: "der Kopf
muss auf positiven Bildern ueber dem LAGEPRIORE liegen, bevor irgendein
Vergleich gerechnet wird". Nicht ueber dem Zufall.

Der Unterschied ist der ganze Punkt. Der Zufallswert der Punkt-AUC ist 0.5 und
gehoert einem Zeiger, der irgendwohin zeigt, auch auf die Luft neben dem
Koerper. Der Lagepriore ist die gemittelte Kastenkarte des Trainingsteils, also
eine Karte, die nichts kann ausser zu wissen, wo Verschattungen ueblicherweise
sitzen. Wer den nicht schlaegt, hat Anatomie gelernt und keine Pathologie.

Und der Test hat einen zweiten Zweck, der billiger ist als er aussieht: er
faengt die Kastenfalle. Wenn Bild und Kasten in der Augmentierung
auseinanderlaufen, faellt der Verlust trotzdem, das Skript laeuft trotzdem
durch, und nur diese Zahl zeigt es. Deshalb laeuft er nach FOLD 0 und nicht
nach Fold 4: er kostet Minuten und kann zweieinhalb verlorene Stunden sparen.

WAS ER NICHT IST
----------------
Er ist KEIN Endpunkt von Phase 5. Der Endpunkt ist A, die geschichtete AUC, und
die Frage ist, was der zweite Kopf KOSTET. Dass ein Modell, das aufs Zeigen
trainiert wurde, besser zeigt als eines, das nie darum gebeten wurde, ist eine
Definition und keine Erkenntnis.

Er beziffert allerdings etwas, das in die Mappe gehoert: der Vorsprung des
EINKOEPFIGEN Modells ueber dem Lagepriore ist das emergente Niveau, der des
zweikoepfigen das antrainierte, und die Differenz ist, was die Aufsicht
gebracht hat. Das einkoepfige Niveau liefert `rsna_cam_power.py` aus Phase 2.

JEDE ZAHL AUFGESCHLUESSELT
--------------------------
Kein Urteil ohne Streuung, und keine Kennzahl ohne die Aufschluesselung je
Projektion daneben. Beides sind Regeln, die dieses Projekt bezahlt hat: in
Phase 3 war jede der fuenf Ueberschriften ein gepoolter Mischwert, und die
Sensitivitaetsluecke zwischen AP und PA (0.882 gegen 0.574) war darin
unsichtbar. Der Kopf koennte in AP fliegen und in PA nichts tun, und gepoolt
saehe das nach Erfolg aus.

CLI, aus dem Repo-Wurzelverzeichnis:
  .\\venv\\Scripts\\python.exe rsna\\befunde\\rsna_kopf_auswertung.py rauchtest ^
      --pred-dir predictions_kopf_exclude --folds 0
  .\\venv\\Scripts\\python.exe rsna\\befunde\\rsna_kopf_auswertung.py rauchtest ^
      --pred-dir predictions_kopf_empty --folds 0 1 2 3 4
"""

from __future__ import annotations

import _repo_path  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_lokalisation import (REF_SIZE, box_mask, evaluate_map, load_boxes,
                               load_lung, to_reference)

BOOT = 2000               # Bootstrap-Ziehungen fuer das Intervall
MIN_N = 30                # unter so vielen Bildern wird kein Urteil gedruckt


# --------------------------------------------------------------------------
# Statistik
# --------------------------------------------------------------------------

def paired_boot(d: np.ndarray, seed: int = 0, boot: int = BOOT) -> dict:
    """Mittlere gepaarte Differenz mit 95-Prozent-Bootstrap-Intervall.

    Gepaart heisst: dasselbe Bild wird von beiden Karten bewertet und die
    Differenz je Bild gebildet. Das entfernt die Streuung, die davon kommt,
    dass manche Faelle einfach schwerer sind als andere, und die ist groesser
    als der gesuchte Effekt.

    Bootstrap und nicht t-Test, weil die Punkt-AUC je Bild schief verteilt ist
    (sie ist nach oben durch 1 begrenzt und viele Bilder liegen nahe dran). Ein
    t-Intervall unterstellt Symmetrie, die hier nicht da ist.

    `n` steht ausdruecklich im Ergebnis. Ein Urteil ohne die Fallzahl daneben
    ist in diesem Projekt schon zweimal danebengegangen.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return {"n": n, "mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "sd": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(boot, n))
    m = d[idx].mean(axis=1)
    return {"n": n, "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
            "lo": float(np.percentile(m, 2.5)),
            "hi": float(np.percentile(m, 97.5))}


# --------------------------------------------------------------------------
# Herkunft
# --------------------------------------------------------------------------

def check_npz(npz, val_ids: list[str]) -> None:
    """Gehoert dieses Feld zu diesem Fold?

    Eine Datei sagt nicht, aus welchem Lauf sie stammt. Einmal sind in diesem
    Projekt fuenf Gewichte unbemerkt ueberschrieben worden, weil der Dateiname
    Fold und Seed trug und sonst nichts. Hier ist der Test billig: die
    gespeicherten patientId muessen genau die Validierungs-IDs des Folds sein,
    in derselben Reihenfolge, in der `predict` sie erzeugt hat.
    """
    got = [str(x) for x in npz["patientId"]]
    if got != list(val_ids):
        raise SystemExit(
            f"ABBRUCH: das Kopffeld enthaelt {len(got)} IDs, der Fold hat "
            f"{len(val_ids)}, und sie stimmen nicht ueberein. Diese Datei "
            f"gehoert zu einem anderen Fold oder einem anderen Split.")


# --------------------------------------------------------------------------

def score_fold(fold: int, args, boxes: dict, sp: dict) -> pd.DataFrame:
    f = Path(args.pred_dir) / f"head_f{fold}_s{args.seed}.npz"
    prior_p = Path(args.baselines) / f"prior_f{fold}.npy"
    if not f.exists():
        print(f"  fehlt: {f}")
        return pd.DataFrame()
    if not prior_p.exists():
        raise SystemExit(
            f"ABBRUCH: {prior_p} fehlt. Der Lagepriore kommt aus Phase 1:\n"
            f"  python rsna\\befunde\\rsna_lokalisation.py tor")

    val_ids = sp["folds"][fold]["val"]
    z = np.load(f, allow_pickle=False)
    check_npz(z, val_ids)
    fields = z["field"]
    prior = np.load(prior_p)
    vpmap = sp["viewpos"]

    rows, no_mask = [], 0
    pos = [(k, i) for k, i in enumerate(val_ids) if i in boxes]
    print(f"\nFold {fold}: {len(pos)} annotierte Validierungsbilder, "
          f"Kopfraster {int(z['grid'])} x {int(z['grid'])}")
    for j, (k, pid) in enumerate(pos, 1):
        lung = load_lung(args.masks, pid)
        if lung is None:
            no_mask += 1
            continue
        b = box_mask(boxes[pid], REF_SIZE)
        # Beide Karten gehen denselben Weg auf das Referenzraster: der Kopf von
        # 14 x 14 bilinear hoch, der Lagepriore liegt schon dort. Waeren die
        # Wege verschieden, verglichen wir zwei Instrumente statt zweier Karten.
        for name, heat in (("Kopf", to_reference(fields[k], REF_SIZE)),
                           ("Lagepriore", prior)):
            r = evaluate_map(heat, b, lung)
            r.update({"fold": fold, "patientId": pid, "map": name,
                      "viewpos": vpmap.get(pid, "?")})
            rows.append(r)
        if j % 250 == 0:
            print(f"    {j}/{len(pos)}")
    if no_mask:
        print(f"  {no_mask} Bilder ohne Lungenmaske uebersprungen")
    return pd.DataFrame(rows)


def verdict(d: pd.DataFrame, label: str, col: str = "point_auc_lung",
            seed: int = 0) -> dict:
    """Kopf gegen Lagepriore auf einer Teilmenge, gepaart je Bild."""
    w = d.pivot_table(index=["fold", "patientId"], columns="map", values=col)
    w = w.dropna()
    if len(w) < MIN_N:
        print(f"  {label:<14} nur {len(w)} Bilder, unter der Mindestzahl "
              f"{MIN_N}. Kein Urteil, das waere Rauschen mit Nachkommastellen.")
        return {}
    r = paired_boot((w["Kopf"] - w["Lagepriore"]).to_numpy(), seed)
    r.update({"gruppe": label, "kopf": float(w["Kopf"].mean()),
              "priore": float(w["Lagepriore"].mean())})
    print(f"  {label:<14} n {r['n']:>5}   Kopf {r['kopf']:.4f}   "
          f"Lagepriore {r['priore']:.4f}   Differenz {r['mean']:+.4f} "
          f"[{r['lo']:+.4f}, {r['hi']:+.4f}]")
    return r


def run(args) -> int:
    sp = json.loads(Path(args.splits).read_text())
    boxes = load_boxes(args.csv)
    out_dir = Path(args.out_dir or args.pred_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = [score_fold(f, args, boxes, sp) for f in args.folds]
    parts = [p for p in parts if not p.empty]
    if not parts:
        print("nichts ausgewertet")
        return 2
    d = pd.concat(parts, ignore_index=True)
    d.to_csv(out_dir / "kopf_vs_priore_per_image.csv", index=False)

    print("\n" + "=" * 78)
    print("KOPF GEGEN LAGEPRIORE, Punkt-AUC innerhalb der Lungenmaske")
    print("=" * 78)
    print("  gepaart je Bild, 95-Prozent-Bootstrap-Intervall der Differenz")
    print()
    res = [verdict(d, "gepoolt", seed=args.seed)]
    # Die Aufschluesselung steht NEBEN der gepoolten Zahl, nicht statt ihrer.
    # Was gepoolt ueberlebt und aufgeschluesselt zerfaellt, war kein Befund.
    #
    # `ungeprueft` sammelt die Projektionen, fuer die es zu wenige Bilder gab.
    # Ohne diese Liste wuerde das Urteil unten "in beiden Projektionen" sagen,
    # obwohl eine gar nicht geprueft wurde, und das waere genau die Sorte Satz,
    # die dieses Projekt sonst jagt.
    ungeprueft = []
    for v in ("AP", "PA"):
        sub = d[d["viewpos"] == v]
        if sub.empty:
            ungeprueft.append(v)
            continue
        r = verdict(sub, f"nur {v}", seed=args.seed)
        (res if r else ungeprueft).append(r if r else v)
    if len(args.folds) > 1:
        print()
        for f in args.folds:
            sub = d[d["fold"] == f]
            if not sub.empty:
                res.append(verdict(sub, f"Fold {f}", seed=args.seed))

    res = [r for r in res if r]
    pd.DataFrame(res).to_csv(out_dir / "kopf_vs_priore_urteil.csv", index=False)

    # ---- Das Tor -------------------------------------------------------
    print("\n" + "=" * 78)
    print("TOR: der Kopf muss ueber dem Lagepriore liegen")
    print("=" * 78)
    pooled = next((r for r in res if r["gruppe"] == "gepoolt"), None)
    if pooled is None:
        print("  kein Urteil moeglich")
        return 2
    strata = [r for r in res if r["gruppe"].startswith("nur ")]

    ok_pool = pooled["lo"] > 0
    bad = [r for r in strata if not (r["lo"] > 0)]
    print(f"  gepoolt: Differenz {pooled['mean']:+.4f} "
          f"[{pooled['lo']:+.4f}, {pooled['hi']:+.4f}]  "
          f"{'ueber' if ok_pool else 'NICHT gesichert ueber'} dem Lagepriore")
    for r in strata:
        print(f"  {r['gruppe']}: {r['mean']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}]")

    if ungeprueft:
        print(f"  ungeprueft, zu wenige Bilder: {', '.join(ungeprueft)}")

    if ok_pool and not bad:
        wo = ("in beiden Projektionen" if not ungeprueft
              else f"in {', '.join(r['gruppe'][4:] for r in strata)}")
        print(f"\n  BESTANDEN. Der Kopf zeigt besser als eine Karte, die nur die")
        print(f"  uebliche Lage von Verschattungen kennt, und zwar {wo}.")
        print("  Die Aufsicht ist angekommen, die Kaesten sind in der")
        print("  Augmentierung nicht verrutscht. Der Vergleich darf laufen.")
        if ungeprueft:
            print(f"  ABER: {', '.join(ungeprueft)} wurde NICHT geprueft. Das ist")
            print("  keine Bestaetigung fuer diese Projektion, nur das Fehlen")
            print("  einer Messung.")
        return 0
    if ok_pool and bad:
        print("\n  NUR GEPOOLT BESTANDEN, und das reicht nicht.")
        for r in bad:
            print(f"  In {r['gruppe'][4:]} ist der Vorsprung nicht gesichert "
                  f"({r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}], "
                  f"n = {r['n']}).")
        print("  Genau diese Form hatte der Fehler in Phase 3: eine Ueberschrift,")
        print("  die haelt, und darunter zwei Projektionen, von denen eine nicht")
        print("  mitkommt. Vor dem Weiterrechnen klaeren.")
        print("  ACHTUNG: 'nicht gesichert' heisst NICHT 'kein Vorsprung'. Bei")
        print("  kleinem n kann es auch heissen, dass zu ungenau gemessen wurde.")
        return 1
    print("\n  DURCHGEFALLEN. Der Kopf liegt nicht gesichert ueber dem Lagepriore.")
    print("  Kein Vergleich rechnen, bevor das geklaert ist. Die drei Verdaechtigen,")
    print("  in der Reihenfolge ihrer Wahrscheinlichkeit:")
    print("    1. Die Kastenfalle. Bild und Maske laufen in der Augmentierung")
    print("       auseinander.  pytest tests/test_rsna_kopf.py")
    print("    2. Der Kopf sagt ueberall Null. Sichtbar an einem sehr niedrigen")
    print("       train_loc_loss bei gleichzeitig flachem Feld; dann ist")
    print("       pos_weight je Kachel zu klein, siehe --head-negatives.")
    print("    3. Lambda zu klein, der Kopfverlust kommt gegen den")
    print("       Klassifikationsverlust nicht an. Steht als head_lambda in")
    print("       results_rsna.csv.")
    return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("rauchtest", help="Kopffeld gegen den Lagepriore")
    g.add_argument("--pred-dir", type=Path, required=True,
                   help="Ordner mit head_f*_s*.npz aus dem Lauf")
    g.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    g.add_argument("--csv", type=Path, default=Path("data/rsna"))
    g.add_argument("--masks", type=Path, default=Path("data/rsna/masks224_dev"))
    g.add_argument("--baselines", type=Path, default=Path("predictions_lokalisation"),
                   help="wo Phase 1 prior_f*.npy abgelegt hat")
    g.add_argument("--folds", type=int, nargs="+", default=[0])
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out-dir", type=Path, default=None)
    g.set_defaults(func=run)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
