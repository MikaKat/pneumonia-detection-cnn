"""External validation of the DELIVERED ensemble on Kermany, inference only.

What this produces
------------------
`predictions_extern_kermany/extern_kermany_ens.csv` with one row per image and
the raw plus calibrated probability of each fold, `bericht.json` with every
number the report quotes, and a verdict on the console against the criteria
written down in `erklaerungen/34_externe_validierung_vorfestlegung.md` before
the first image was read.

Why it exists
-------------
The external number this project quotes, 0.885 leak adjusted, belongs to a
SINGLE checkpoint from an earlier stage. What is delivered is something else: an
ensemble of five two headed models, each with its own Platt curve, averaged as
probabilities, with a threshold of 0.2003. That object has never seen a foreign
dataset. "Does it generalise" is the most likely question a reviewer asks, and
the honest answer today is "unknown for the thing that ships".

The second reason is methodical. The RSNA holdout was spent on 09.08.2026 and
gives no unbiased number to any later change. A foreign dataset is not bound by
that: nothing in it was ever fitted, tuned or reported, so it is a fresh
measurement rather than a second look at a used one.

How to read the result
----------------------
The primary number is the leak adjusted external AUC of the calibrated ensemble
under the preprocessing that ships. AUC is the c statistic: the chance that a
random pneumonia image is scored above a random normal one, 0.5 being chance.
It is POOLED, not stratified by projection, because Kermany carries no
ViewPosition. Every stratified number of this project (A) is therefore NOT
directly comparable, and neither is the holdout value 0.8687.

The reference values to read it against are printed above it: the metadata leak
of this dataset (image dimensions alone separate the classes at about 0.91),
the internal cross validation 0.8368, and the earlier single model external
value 0.885.

Calibration and threshold are reported separately from discrimination, because
they travel separately. A model can keep the ordering intact and still put the
wrong probability on it, and that is what a prevalence jumping from 0.225 to
0.73 does to a Platt curve fitted at the lower one.

What would refute a positive reading: a leak adjusted AUC near 0.5, which would
mean the raw number was the file geometry rather than the lungs, or a large gap
between the 'pad' and 'stretch' variants, which would mean the result is an
artefact of squashing non square images.

What this cannot answer
-----------------------
1. The confounder C, the AUC of the score against the projection. Kermany has
   no projection annotation. C stays measured on RSNA alone.
2. The localisation head. Kermany has no boxes. The head is computed and stored
   so that its mean activation can be inspected, but there is nothing to score
   it against.
3. Anything about adults. Kermany is paediatric, ages one to five.

CLI:
  venv\\Scripts\\python.exe rsna\\befunde\\rsna_extern_kermany_ens.py --dml-index 1
  ... --dml-index 1 --split test       # nur der offizielle Testteil
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

import _repo_path  # noqa: F401  (puts the neighbour folders on the path)

from rsna_external_kermany import (build_variants, collect_kermany,
                                   grouped_bootstrap_auc, header_leak,
                                   operating_point, read_dims,
                                   stratified_by_score)
from rsna_platt import brier, ece

ARM_TAG = "_p5head_ex"
FOLDS = [0, 1, 2, 3, 4]
PRIMAER_VARIANTE = "stretch"

# ---- pre-registered, see erklaerungen/34_externe_validierung_vorfestlegung.md
TOR_AUC = 0.80            # primary: leak adjusted external AUC at or above this
GRAUZONE_AUC = 0.75       # between the two: transfers, weaker than before
TOR_ECE = 0.10            # calibration counts as carried below this
ERWARTUNG = {
    "auc_leak_bereinigt": "0,84 bis 0,90, unterhalb der alten 0,885",
    "ece_kalibriert": "UEBER 0,20, die Kalibrierung traegt voraussichtlich NICHT",
    "sens_bei_schwelle": "ueber 0,90",
    "npv_bei_schwelle": "ueber 0,60, deutlich besser als die alten 0,500",
    "spez_bei_schwelle": "unter 0,60",
    "pad_minus_stretch": "zwischen -0,01 und +0,01, wie im Einzelmodell",
}


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


def lade_kalibrierung(pfad: Path) -> tuple[dict, dict, float]:
    if not pfad.is_file():
        abbruch(f"{pfad} fehlt. Ohne Kurve und Schwelle gibt es kein "
                f"ausgeliefertes Modell, nur Rohwerte.")
    kal = json.loads(pfad.read_text(encoding="utf-8"))
    if kal.get("arm") != ARM_TAG:
        abbruch(f"die Kalibrierdatei gehoert zu Arm {kal.get('arm')!r}, "
                f"gebraucht wird {ARM_TAG!r}")
    kurven = {int(e["fold"]): (float(e["a"]), float(e["b"])) for e in kal["platt"]}
    if sorted(kurven) != FOLDS:
        abbruch(f"die Kalibrierdatei kennt die Folds {sorted(kurven)}, "
                f"gebraucht werden {FOLDS}")
    return kal, kurven, float(kal["schwelle"])


def lade_modelle(device):
    """The five delivered weights, with both heads checked rather than assumed.

    A single headed checkpoint would load without complaint under
    `strict=False` and would then quietly deliver a different model. That has
    happened in this project once already, which is why the check is here and
    not in a comment.
    """
    import torch

    from rsna_train import HEAD_GRID, make_model

    modelle, pruef = [], {}
    for k in FOLDS:
        ck = Path("checkpoints") / f"rsna_f{k}_s0_p5head_ex.pth"
        if not ck.is_file():
            abbruch(f"{ck} fehlt")
        pruef[ck.name] = pruefsumme(ck)
        m = make_model(device, head=True, grid=HEAD_GRID)
        state = torch.load(str(ck), map_location="cpu", weights_only=True)
        fehlt, zuviel = m.load_state_dict(state, strict=False)
        if fehlt or zuviel:
            abbruch(f"{ck.name} passt nicht auf das zweikoepfige Modell.\n"
                    f"         fehlend {list(fehlt)[:4]}, ueberzaehlig "
                    f"{list(zuviel)[:4]}")
        m.eval()
        modelle.append(m)
        print(f"    {ck.name}  sha {pruef[ck.name]}")
    return modelle, pruef


def rechne(pfade, tf, modelle, device, batch: int, workers: int):
    """One pass over the images, all five models per batch.

    Five passes would decode every JPEG five times, and decoding costs more
    here than the forward pass of a ResNet18.

    Returns the raw probabilities [n, 5] and the mean head activation [n, 5].
    """
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader

    class ListDataset:
        def __init__(self, paths, tf):
            self.paths, self.tf = paths, tf

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            return self.tf(Image.open(self.paths[i]).convert("L"))

    dl = DataLoader(ListDataset(list(pfade), tf), batch_size=batch,
                    num_workers=workers)
    p_out = [[] for _ in modelle]
    f_out = [[] for _ in modelle]
    with torch.no_grad():
        for bi, x in enumerate(dl, 1):
            x = x.to(device)
            for k, m in enumerate(modelle):
                logits, feld = m(x)
                p_out[k].append(torch.sigmoid(logits.squeeze(1)).float().cpu().numpy())
                f_out[k].append(torch.sigmoid(feld[:, 0]).float().mean(
                    dim=(1, 2)).cpu().numpy())
            if bi % 25 == 0:
                print(f"      Stapel {bi}/{len(dl)}")
    return (np.stack([np.concatenate(o) for o in p_out], axis=1),
            np.stack([np.concatenate(o) for o in f_out], axis=1))


def zuverlaessigkeit(y, p, faecher: int = 10) -> list[dict]:
    """The reliability table behind the ECE, printed rather than only summed.

    A single ECE hides WHERE the model is wrong. On a dataset whose prevalence
    is three times the one the curve was fitted on, the interesting part is the
    lower half of the scale, where almost every bin will hold more positives
    than its own probability claims.
    """
    kanten = np.linspace(0.0, 1.0, faecher + 1)
    idx = np.digitize(p, kanten[1:-1])
    zeilen = []
    for b in range(faecher):
        m = idx == b
        if not m.any():
            continue
        zeilen.append({"von": float(kanten[b]), "bis": float(kanten[b + 1]),
                       "n": int(m.sum()), "p_mittel": float(p[m].mean()),
                       "anteil_positiv": float(y[m].mean())})
    return zeilen


def urteil(name: str, wert: float, tor: float, richtung: str = ">=") -> str:
    ok = wert >= tor if richtung == ">=" else wert <= tor
    return f"  {name:<44}{wert:>9.4f}  {'BESTANDEN' if ok else 'NICHT BESTANDEN'}"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, default=Path("data/chest_xray"))
    p.add_argument("--split", nargs="*", default=None,
                   help="nur diese Originalteile (train/val/test); leer = alle")
    p.add_argument("--kalibrierung", type=Path,
                   default=Path("serving") / "model" / "kalibrierung_p10.json")
    p.add_argument("--out-dir", type=Path, default=Path("predictions_extern_kermany"))
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--dml-index", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 78)
    print("EXTERNE VALIDIERUNG DES AUSGELIEFERTEN ENSEMBLES")
    print("=" * 78)
    print("  Vorfestlegung: erklaerungen/34_externe_validierung_vorfestlegung.md")
    print(f"  primaere Variante {PRIMAER_VARIANTE!r}, Tor {TOR_AUC:.2f}, "
          f"Grauzone ab {GRAUZONE_AUC:.2f}")
    print("  Erwartungen, VOR dem Lauf aufgeschrieben:")
    for k, v in ERWARTUNG.items():
        print(f"    {k:<22} {v}")

    kal, kurven, schwelle = lade_kalibrierung(args.kalibrierung)
    print(f"\n  Kalibrierung aus {args.kalibrierung}")
    print(f"    Schwelle {schwelle:.4f}, {kal['schwelle_herkunft']}")
    print(f"    intern: ECE roh {kal['dev']['ece_roh']:.4f} -> kalibriert "
          f"{kal['dev']['ece_kal']:.4f} auf {kal['dev']['n']} Bildern, "
          f"Praevalenz {kal['dev']['praevalenz']:.4f}")

    d = collect_kermany(args.images, args.split)
    if d.empty:
        abbruch(f"keine Bilder unter {args.images}")
    y = d.label.values.astype(float)
    gruppen = d.group.values
    print(f"\n  Kermany: {len(d)} Bilder, {d.group.nunique()} Patientengruppen, "
          f"Praevalenz {y.mean():.4f}")
    print(f"    je Originalteil: {dict(d.split.value_counts())}")
    print(f"    Die Praevalenz steigt von {kal['dev']['praevalenz']:.4f} auf "
          f"{y.mean():.4f}, also um Faktor {y.mean() / kal['dev']['praevalenz']:.1f}.")

    # ---- Kontrolle 1: der Metadaten-Leak, VOR jeder Modellzahl -------------
    masse = read_dims(d.path.tolist())
    leak_auc, leak_score = header_leak(masse, y, gruppen, args.seed)
    print(f"\n  Metadaten-Leak (nur Bildabmessungen, gruppierte CV): "
          f"AUC {leak_auc:.4f}")
    print("    Jede Modellzahl unten wird gegen diesen Wert gelesen.")

    from rsna_train import pick_device

    device, _, dev_label = pick_device(args.device, args.dml_index)
    print(f"\n  Hardware: {dev_label}")
    modelle, pruef = lade_modelle(device)

    # ---- rechnen ----------------------------------------------------------
    t_all = time.time()
    roh, feld, kalibriert = {}, {}, {}
    for name, tf in build_variants(args.size).items():
        print(f"\n  Variante {name!r} ...")
        t0 = time.time()
        P, F = rechne(d.path.tolist(), tf, modelle, device, args.batch,
                      args.workers)
        K = np.column_stack([platt_apply(P[:, i], *kurven[k])
                             for i, k in enumerate(FOLDS)])
        roh[name], feld[name], kalibriert[name] = P, F, K
        print(f"    {time.time() - t0:.0f} s, roh im Mittel {P.mean():.4f} "
              f"-> kalibriert {K.mean():.4f}")

    for name in roh:
        for i, k in enumerate(FOLDS):
            d[f"p_{name}_roh_f{k}"] = roh[name][:, i]
            d[f"p_{name}_kal_f{k}"] = kalibriert[name][:, i]
            d[f"kopf_{name}_f{k}"] = feld[name][:, i]
        d[f"p_{name}_ens"] = kalibriert[name].mean(axis=1)
        d[f"p_{name}_ens_roh"] = roh[name].mean(axis=1)
    for c in masse.columns:
        d[c] = masse[c].values
    d["leak_score"] = leak_score
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ziel = args.out_dir / "extern_kermany_ens.csv"
    d.to_csv(ziel, index=False)

    # ---- Bericht ----------------------------------------------------------
    from sklearn.metrics import roc_auc_score

    print("\n" + "=" * 78)
    print("ERGEBNIS")
    print("=" * 78)
    print(f"  intern zum Vergleich: Kreuzvalidierung A 0,8368, Holdout A 0,8687")
    print(f"  frueher, EINZELMODELL extern leak-bereinigt: 0,885")
    print(f"  Metadaten-Leak dieses Datensatzes:           {leak_auc:.4f}")
    print()

    bericht = {
        "wann": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arm": ARM_TAG,
        "images": str(args.images).replace("\\", "/"),
        "split": args.split or "alle",
        "n": int(len(d)),
        "gruppen": int(d.group.nunique()),
        "praevalenz": float(y.mean()),
        "leak_auc": float(leak_auc),
        "schwelle": schwelle,
        "checkpoints": pruef,
        "primaere_variante": PRIMAER_VARIANTE,
        "tor_auc": TOR_AUC,
        "erwartung": ERWARTUNG,
        "varianten": {},
    }

    for name in ("stretch", "pad"):
        K = kalibriert[name]
        ens = K.mean(axis=1)
        je_fold = [float(roc_auc_score(y, K[:, i])) for i in range(K.shape[1])]
        a, lo, hi = grouped_bootstrap_auc(y, ens, gruppen, B=args.bootstrap,
                                          seed=args.seed)
        s_auc, je_quintil = stratified_by_score(y, ens, leak_score)
        marke = "  <- primaer" if name == PRIMAER_VARIANTE else ""
        print(f"  {name:<8} Ensemble {a:.4f} [{lo:.4f}, {hi:.4f}]   "
              f"leak-bereinigt {s_auc:.4f}{marke}")
        print(f"           Einzelfolds {np.mean(je_fold):.4f} "
              f"+- {np.std(je_fold, ddof=1):.4f}   "
              f"({', '.join(f'{v:.4f}' for v in je_fold)})")
        print(f"           Gewinn des Ensembles ueber das Mittel der Folds "
              f"{a - float(np.mean(je_fold)):+.4f}")
        bericht["varianten"][name] = {
            "auc": a, "auc_lo": lo, "auc_hi": hi,
            "auc_leak_bereinigt": float(s_auc),
            "auc_je_fold": je_fold,
            "auc_roh_ensemble": float(roc_auc_score(y, roh[name].mean(axis=1))),
            "je_quintil": [
                {"q": q, "n": n, "n_pos": npos, "n_neg": nneg,
                 "disk_paare": paare, "auc": (None if np.isnan(av) else float(av))}
                for q, n, npos, nneg, paare, av in je_quintil],
        }

    diff = (bericht["varianten"]["pad"]["auc"]
            - bericht["varianten"]["stretch"]["auc"])
    bericht["pad_minus_stretch"] = float(diff)
    print(f"\n  'pad' minus 'stretch': {diff:+.4f}")
    print("    Ein grosser Betrag hiesse: das Ergebnis haengt am Verzerren "
          "nicht quadratischer Bilder.")
    print("    Die primaere Zahl wird NICHT getauscht, auch wenn 'pad' besser ist.")

    ens = kalibriert[PRIMAER_VARIANTE].mean(axis=1)
    print("\n" + "-" * 78)
    print(f"Aufschluesselung nach dem Leak (Variante {PRIMAER_VARIANTE!r})")
    print("-" * 78)
    print(f"  {'Q':<3}{'n':>7}{'n_pos':>8}{'n_neg':>8}{'disk.Paare':>13}{'AUC':>9}")
    for q, n, npos, nneg, paare, av in stratified_by_score(y, ens, leak_score)[1]:
        txt = f"{av:.4f}" if not np.isnan(av) else "    n/a"
        print(f"  {q:<3}{n:>7}{npos:>8}{nneg:>8}{paare:>13}{txt:>9}")
    print("    Gewichtet nach diskordanten Paaren, nicht nach n. Ein fast "
          "reines Quintil")
    print("    ist gross und traegt trotzdem kaum Information.")

    # ---- Kalibrierung ------------------------------------------------------
    print("\n" + "-" * 78)
    print("Traegt die Kalibrierung?")
    print("-" * 78)
    e = ece(y, ens)
    b = brier(y, ens)
    print(f"  ECE {e:.4f} gegen intern {kal['dev']['ece_kal']:.4f}, "
          f"Brier {b:.4f} gegen intern {kal['dev']['brier_kal']:.4f}")
    print(f"  {'Fach':<14}{'n':>7}{'p im Mittel':>14}{'tatsaechlich':>14}"
          f"{'Abstand':>10}")
    tabelle = zuverlaessigkeit(y, ens)
    for z in tabelle:
        print(f"  {z['von']:.1f} bis {z['bis']:.1f}   {z['n']:>7}"
              f"{z['p_mittel']:>14.4f}{z['anteil_positiv']:>14.4f}"
              f"{z['anteil_positiv'] - z['p_mittel']:>+10.4f}")
    print("  Eine Kurve, die bei der einen Praevalenz angepasst wurde, kann bei "
          "einer")
    print("  dreifach hoeheren nicht stimmen. Erwartet war ein grosser ECE.")
    bericht["ece"] = float(e)
    bericht["brier"] = float(b)
    bericht["zuverlaessigkeit"] = tabelle

    # ---- Schwelle ----------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"Traegt die Schwelle {schwelle:.4f}?")
    print("-" * 78)
    op = operating_point(y, ens, schwelle)
    print(f"  Sens {op['sens']:.4f} | Spez {op['spec']:.4f} | "
          f"PPV {op['ppv']:.4f} | NPV {op['npv']:.4f}")
    print(f"  als positiv eingestuft {op['pos_rate']:.4f} gegen "
          f"tatsaechlich {y.mean():.4f}")
    print(f"  intern bei derselben Schwelle: Sens "
          f"{kal['dev_bei_schwelle']['sens']:.4f} | Spez "
          f"{kal['dev_bei_schwelle']['spez']:.4f}")
    print("  Der NPV ist die Zahl, auf die es klinisch ankommt: wie oft ein "
          "'unauffaellig'")
    print("  wirklich unauffaellig war. Beim Einzelmodell lag er bei 0,500.")
    bericht["arbeitspunkt"] = op

    # ---- Kopf --------------------------------------------------------------
    kopf = feld[PRIMAER_VARIANTE].mean(axis=1)
    print(f"\n  Kopffeld im Mittel {kopf.mean():.4f} "
          f"(positiv {kopf[y == 1].mean():.4f}, negativ {kopf[y == 0].mean():.4f}). "
          f"Ohne Kaesten NICHT bewertbar, nur abgelegt.")
    bericht["kopf_mittel"] = {"alle": float(kopf.mean()),
                              "positiv": float(kopf[y == 1].mean()),
                              "negativ": float(kopf[y == 0].mean())}

    # ---- das vorfestgelegte Urteil ----------------------------------------
    prim = bericht["varianten"][PRIMAER_VARIANTE]["auc_leak_bereinigt"]
    print("\n" + "=" * 78)
    print("URTEIL nach den vorher aufgeschriebenen Kriterien")
    print("=" * 78)
    print(urteil("primaer: AUC leak-bereinigt", prim, TOR_AUC))
    if GRAUZONE_AUC <= prim < TOR_AUC:
        print("    GRAUZONE: der Transfer traegt, aber schwaecher als das "
              "Einzelmodell.")
    print(urteil("Kalibrierung: ECE", e, TOR_ECE, "<="))
    print(f"  Das Ergebnis beruehrt den verbrauchten Holdout nicht. Es ist eine "
          f"neue Messung.")
    bericht["urteil"] = {
        "primaer_wert": prim, "primaer_tor": TOR_AUC,
        "primaer_bestanden": bool(prim >= TOR_AUC),
        "grauzone": bool(GRAUZONE_AUC <= prim < TOR_AUC),
        "ece_bestanden": bool(e <= TOR_ECE),
    }
    bericht["dauer_s"] = round(time.time() - t_all, 1)

    (args.out_dir / "bericht.json").write_text(
        json.dumps(bericht, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\n  geschrieben: {ziel}")
    print(f"               {args.out_dir / 'bericht.json'}")
    print(f"  Gesamtdauer {time.time() - t_all:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
