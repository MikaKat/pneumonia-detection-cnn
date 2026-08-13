"""External validation of the DELIVERED ensemble on VinDr-CXR, inference only.

What this produces
------------------
`predictions_extern_vindr/extern_vindr_ens.csv` with one row per image, the raw
and calibrated probability of each fold, and the localisation measures against
the radiologist boxes. `bericht.json` carries every number the report quotes and
a verdict against the criteria written down in
`erklaerungen/39_extern_vindr_vorfestlegung.md` before the first image was read.

Why this exists, and why it is NOT the Kermany run again
--------------------------------------------------------
Kermany answered half the question: the ordering survived (0.9231 leak
adjusted), the probability did not (ECE 0.4783). Two things it could not touch,
because it does not contain them: the localisation head, for lack of boxes, and
adults.

VinDr supplies both. 1588 of its 15000 images carry radiologist boxes for the
opacity family, which is the closest public match to RSNA's target that exists:
RSNA's label is literally "lung opacity consistent with pneumonia", drawn as a
box. So the head field, 0.9123 internally against a 0.7520 position prior, gets
its first foreign measurement.

WHAT IT CANNOT ANSWER, said here rather than discovered later
--------------------------------------------------------------
C, the AUC of the score against the projection. The 512 px release is PNG
without DICOM headers, so there is no ViewPosition, and the published set is
stated to be PA only anyway. C stays measured on RSNA alone and waits for
PadChest-GR.

The pre-registered readings
---------------------------
Everything below was fixed in `39_` before any inference:

  target      Lung Opacity OR Consolidation OR Infiltration
  negative    none of those three, INCLUDING the middle class. That is RSNA's
              task: there, 44 % of images are abnormal without opacity.
  primary     point AUC of the head field against the boxes, gate 0.75, which
              is the position prior. Below it the pointer is not a pointer.
  secondary   A, leak adjusted, gate 0.70.
  leak        0.7822, already computed by `rsna_vindr_vorpruefung.py` from the
              image dimensions in the CSV. Not recomputed here, not revised.

Boxes come in ORIGINAL image coordinates. They are mapped through width/height
onto the BOX_SPACE 1024 grid and then through `box_mask` unchanged, so the path
into the 224 reference grid is bit for bit the RSNA one.

Bootstrap resamples IMAGES, not patient groups. In VinDr every image is its own
patient, so this is the one run in this project where that is correct, and the
reason is written down rather than assumed.

CLI (from the repo root):
  venv\\Scripts\\python.exe rsna\\befunde\\rsna_extern_vindr_ens.py --dml-index 1
  ... --limit 300              # Rauchtest auf wenigen Bildern
  ... --lesart K               # die vorher angemeldete Nebenrechnung
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

import _repo_path  # noqa: F401

from rsna_lokalisation import REF_SIZE, box_mask, evaluate_map, load_lung
# ACHTUNG, die Reihenfolge: rsna_platt.ece(y, p) und brier(y, p) nehmen das
# LABEL zuerst. Dieses Skript rief sechsmal ece(p, y) auf, und weil `ece` nach
# dem ERSTEN Argument in Faecher teilt, wurde damit nach dem Label statt nach
# der Wahrscheinlichkeit gefaechert. Berichtet wurden 0,1584 statt 0,0239.
# `brier` ist symmetrisch und war zufaellig richtig, was den Fehler verdeckt
# hat: eine der beiden Zahlen stimmte. Gefunden am 13.08.2026 beim Nachrechnen
# fuer die Lesart M.
from rsna_platt import brier, ece

ARM_TAG = "_p5head_ex"
FOLDS = [0, 1, 2, 3, 4]
BOX_SPACE = 1024
ZIEL = ["Lung Opacity", "Consolidation", "Infiltration"]

# ---- vorfestgelegt, siehe erklaerungen/39_extern_vindr_vorfestlegung.md
TOR_KOPF = 0.75           # primary: the position prior of phase 5
TOR_A = 0.70              # secondary: leak adjusted A
# Aus rsna_vindr_vorpruefung.py, je Lesart. Nur zum Gegenpruefen, nicht zum
# Rechnen: die Zahl im Bericht kommt immer aus dem Lauf selbst.
LECK = {"A": 0.7822, "K": 0.7828, "M": 0.7677}
INTERN_KOPF = 0.9123
INTERN_A = 0.8368


def abbruch(text: str) -> None:
    print(f"\n  ABBRUCH: {text}\n", file=sys.stderr)
    raise SystemExit(2)


def pruefsumme(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def platt_apply(p, a, b):
    return 1.0 / (1.0 + np.exp(-(a * logit(p) + b)))


# --------------------------------------------------------------- Kalibrierung

def lade_kalibrierung(pfad: Path):
    if not pfad.is_file():
        abbruch(f"{pfad} fehlt. Ohne Kurve und Schwelle gibt es kein "
                f"ausgeliefertes Modell, nur Rohwerte.")
    kal = json.loads(pfad.read_text(encoding="utf-8"))
    if kal.get("arm") != ARM_TAG:
        abbruch(f"die Kalibrierdatei gehoert zu Arm {kal.get('arm')!r}, "
                f"gebraucht wird {ARM_TAG!r}")
    kurven = {int(e["fold"]): (float(e["a"]), float(e["b"])) for e in kal["platt"]}
    if sorted(kurven) != FOLDS:
        abbruch(f"die Kalibrierdatei kennt die Folds {sorted(kurven)}")
    return kal, kurven, float(kal["schwelle"])


# --------------------------------------------------------------- Daten

def sammle(bilder: Path, csv: Path, lesart: str):
    """Ein Eintrag je Bild: Pfad, Ziel-Label, Kaesten je Radiologe.

    Die Lesarten stehen in `39_`. A ist primaer, K die angemeldete
    Nebenrechnung; B wird bewusst nicht angeboten.
    """
    d = pd.read_csv(csv)
    for sp in ["image_id", "class_name", "rad_id", "width", "height"]:
        if sp not in d.columns:
            abbruch(f"{csv} hat keine Spalte {sp!r}. Gebraucht wird die CSV des "
                    f"512er-Datensatzes, nicht die amtliche des Wettbewerbs.")

    ziel = d[d.class_name.isin(ZIEL)].dropna(subset=["x_min"])
    zust = ziel.groupby("image_id")["rad_id"].nunique()
    masse = d.groupby("image_id")[["width", "height"]].first()

    eintraege = []
    for iid, (w, h) in masse.iterrows():
        p = bilder / f"{iid}.png"
        if not p.is_file():
            continue
        n_rad = int(zust.get(iid, 0))
        if lesart == "A":
            y = 1 if n_rad >= 1 else 0
        elif lesart == "K":
            if n_rad == 1:
                continue                      # Einzelmeinungen fallen raus
            y = 1 if n_rad >= 2 else 0
        elif lesart == "M":
            # Strenge Mehrheit: sagen zwei von drei nein, ist es nein.
            # NACHTRAEGLICH hinzugefuegt am 13.08.2026 auf Mikas Festlegung,
            # nachdem das Ergebnis unter A bekannt war. Siehe erklaerungen/43_.
            # Der Unterschied zu K ist nicht kosmetisch: K sagt "wir wissen es
            # nicht" und wirft die 750 weg, M sagt "sie sind negativ".
            y = 1 if n_rad >= 2 else 0
        else:
            abbruch(f"Lesart {lesart!r} ist nicht vorgesehen. A, K oder M.")
        eintraege.append({"image_id": iid, "path": str(p), "y": y,
                          "n_rad": n_rad, "width": float(w), "height": float(h)})
    if not eintraege:
        abbruch(f"keine Bilder unter {bilder} gefunden")
    return pd.DataFrame(eintraege), ziel


def kaesten_je_radiologe(ziel: pd.DataFrame, iid: str, w: float, h: float):
    """Kaesten dieses Bildes, je Radiologe, im BOX_SPACE-Raster.

    Original -> 1024er Raster ueber die ECHTEN Bildmasse, danach `box_mask`
    unveraendert. Das ist derselbe Weg wie bei RSNA, wo die Kaesten schon in
    1024 vorliegen, und deshalb steht die Zahl neben der internen.
    """
    sub = ziel[ziel.image_id == iid]
    aus = {}
    sw, sh = BOX_SPACE / w, BOX_SPACE / h
    for rad, g in sub.groupby("rad_id"):
        aus[rad] = [(float(r.x_min) * sw, float(r.y_min) * sh,
                     float(r.x_max - r.x_min) * sw, float(r.y_max - r.y_min) * sh)
                    for r in g.itertuples()]
    return aus


# --------------------------------------------------------------- Modell

def lade_modelle(device):
    """Die fuenf ausgelieferten Gewichte, beide Koepfe geprueft statt vermutet."""
    import torch

    from rsna_train import HEAD_GRID, make_model

    modelle, pruef = [], {}
    for k in FOLDS:
        ck = Path("checkpoints") / f"rsna_f{k}_s0{ARM_TAG}.pth"
        if not ck.is_file():
            abbruch(f"{ck} fehlt")
        pruef[ck.name] = pruefsumme(ck)
        m = make_model(device, head=True, grid=HEAD_GRID)
        state = torch.load(str(ck), map_location="cpu", weights_only=True)
        fehlt, zuviel = m.load_state_dict(state, strict=False)
        if fehlt or zuviel:
            abbruch(f"{ck.name} passt nicht auf das zweikoepfige Modell.")
        m.eval()
        modelle.append(m)
        print(f"    {ck.name}  sha {pruef[ck.name]}")
    return modelle, pruef


def baue_kette(size: int):
    """Die AUSGELIEFERTE Kette. Entfaerben VOR dem Skalieren.

    Die umgekehrte Reihenfolge war der Fehler, der beim Aufschreiben des
    Rauchtests gefunden wurde. Auf grauen Bildern null Unterschied, aber die
    Reihenfolge steht hier trotzdem richtig, weil sie sonst wieder verrutscht.

    'pad' entfaellt: die 512er-PNG sind bereits auf ein Quadrat gezogen, ein
    Zurueckziehen waere ein zweiter Abtastschritt. Siehe `39_`, Abschnitt
    Vorverarbeitung.
    """
    import torchvision.transforms as T

    from rsna_train import IMNET_MEAN, IMNET_STD

    return T.Compose([T.Grayscale(num_output_channels=3),
                      T.Resize((size, size)),
                      T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)])


class ListDataset:
    """AUF MODULEBENE, und das ist kein Stilentscheid.

    Windows startet die DataLoader-Arbeiter mit `spawn`, nicht mit `fork`. Der
    Datensatz muss dafuer gepickelt und im Kindprozess wieder aufgebaut werden,
    und eine Klasse, die INNERHALB einer Funktion definiert ist, laesst sich
    nicht pickeln: `Can't pickle local object 'rechne.<locals>.ListDataset'`.

    Der Kermany-Lauf hat dieselbe Klasse verschachtelt und ist trotzdem
    durchgelaufen, weil dort `--workers 0` steht und dann gar kein Kindprozess
    entsteht. Der Fehler lag also latent daneben und wurde erst sichtbar, als
    die Vorgabe hochgesetzt wurde. Hier steht sie jetzt richtig, und damit sind
    Arbeiter ueberhaupt erst benutzbar.
    """

    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        return self.tf(Image.open(self.paths[i]).convert("L"))


def rechne(pfade, tf, modelle, device, batch: int, workers: int):
    """Ein Durchgang, alle fuenf Modelle je Stapel.

    Anders als beim Kermany-Lauf wird das GANZE Kopffeld behalten, nicht nur
    sein Mittelwert: hier gibt es Kaesten, gegen die es geprueft werden kann.
    """
    import torch
    from torch.utils.data import DataLoader

    dl = DataLoader(ListDataset(list(pfade), tf), batch_size=batch,
                    num_workers=workers)
    p_out = [[] for _ in modelle]
    felder = []
    with torch.no_grad():
        for bi, x in enumerate(dl, 1):
            x = x.to(device)
            stapel = []
            for k, m in enumerate(modelle):
                logits, feld = m(x)
                p_out[k].append(torch.sigmoid(logits.squeeze(1)).float().cpu().numpy())
                stapel.append(torch.sigmoid(feld[:, 0]).float().cpu().numpy())
            felder.append(np.stack(stapel, axis=1))     # [b, 5, g, g]
            if bi % 25 == 0:
                print(f"      Stapel {bi}/{len(dl)}")
    return (np.stack([np.concatenate(o) for o in p_out], axis=1),
            np.concatenate(felder, axis=0))


def feld_auf_referenz(feld_gg: np.ndarray) -> np.ndarray:
    """14x14 bilinear auf 224. Genau wie in `rsna_kopf_auswertung.score_fold`."""
    import torch
    t = torch.from_numpy(feld_gg)[None, None].float()
    up = torch.nn.functional.interpolate(t, size=(REF_SIZE, REF_SIZE),
                                         mode="bilinear", align_corners=False)
    return up[0, 0].numpy()


# --------------------------------------------------------------- Kennzahlen

def boot_auc(y, p, B=500, seed=0):
    """Bootstrap ueber BILDER. Bei VinDr ist jedes Bild ein eigener Patient,
    deshalb ist das hier richtig und nur hier."""
    from sklearn.metrics import roc_auc_score
    punkt = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)
    aus = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        aus.append(roc_auc_score(y[idx], p[idx]))
    return punkt, float(np.percentile(aus, 2.5)), float(np.percentile(aus, 97.5))


def boot_mittel(werte, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    w = np.asarray(werte, float)
    w = w[~np.isnan(w)]
    if len(w) < 2:
        return float("nan"), float("nan"), float("nan")
    zieh = rng.choice(w, (B, len(w)), replace=True).mean(axis=1)
    return float(w.mean()), float(np.percentile(zieh, 2.5)), \
        float(np.percentile(zieh, 97.5))


def arbeitspunkt(y, p, thr):
    tp = int(((p >= thr) & (y == 1)).sum())
    fp = int(((p >= thr) & (y == 0)).sum())
    fn = int(((p < thr) & (y == 1)).sum())
    tn = int(((p < thr) & (y == 0)).sum())
    return {"sens": tp / max(tp + fn, 1), "spez": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1),
            "pos_rate": float((p >= thr).mean())}


def zuverlaessigkeit(p, y, bins=10):
    kanten = np.linspace(0, 1, bins + 1)
    aus = []
    for lo, hi in zip(kanten[:-1], kanten[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        aus.append({"von": float(lo), "bis": float(hi), "n": int(m.sum()),
                    "p_mittel": float(p[m].mean()),
                    "anteil_positiv": float(y[m].mean())})
    return aus


def leck_score(d):
    """Der Leck-Score aus den ORIGINALMASSEN, gruppiert kreuzvalidiert.

    Wortgleich mit `header_leak` aus `rsna_external_kermany.py` und mit
    `rsna_vindr_vorpruefung.py`. Er wird hier NEU gerechnet statt eingelesen,
    damit die Schichtung und die Vorhersagen garantiert dieselbe Zeilenfolge
    haben; der Wert muss auf drei Stellen mit der Vorpruefung uebereinstimmen,
    und das Skript sagt es, wenn nicht.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    X = np.column_stack([d.width, d.height, d.width / d.height,
                         d.width * d.height])
    y = d.y.to_numpy()
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True,
                                       random_state=0).split(X, y, d.image_id):
        m = GradientBoostingClassifier(random_state=0).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof)), oof


def leck_bereinigt(y, p, leck, q=5):
    """AUC innerhalb der Quintile des Leck-Scores, GEWICHTET NACH DISKORDANTEN
    PAAREN.

    Nicht nach n. Der Unterschied ist bei Kermany der zwischen 0,853 und 0,885
    gewesen: ein Quintil kann fast rein positiv sein, seine AUC ruht dann auf
    Paaren, die alle dasselbe eine negative Bild enthalten, und nach n gewichtet
    geht dieses Rauschen mit vollem Gewicht ein.
    """
    from sklearn.metrics import roc_auc_score

    kanten = np.quantile(leck, np.linspace(0, 1, q + 1))
    kanten[-1] += 1e-9
    je, num, den = [], 0.0, 0
    for i in range(q):
        s = (leck >= kanten[i]) & (leck < kanten[i + 1])
        npos = int(y[s].sum())
        nneg = int(s.sum()) - npos
        paare = npos * nneg
        if paare == 0 or s.sum() < 30:
            je.append({"q": i + 1, "n": int(s.sum()), "n_pos": npos,
                       "n_neg": nneg, "paare": paare, "auc": None})
            continue
        a = float(roc_auc_score(y[s], p[s]))
        je.append({"q": i + 1, "n": int(s.sum()), "n_pos": npos, "n_neg": nneg,
                   "paare": paare, "auc": a})
        num += a * paare
        den += paare
    return (num / den if den else float("nan")), je


def bbse(p, schwelle, sens, spez):
    """Lipton et al. 2018. Gibt den ROHEN Wert zurueck, nicht den beschnittenen.

    Ein negativer Rohwert ist kein Rundungsfehler: er heisst, dass weniger
    Bilder positiv eingestuft werden als die Quelle allein an Falschalarmen
    erzeugt. Unter Label Shift ist das unmoeglich, und genau deshalb gehoert
    die unbeschnittene Zahl in den Bericht.
    """
    q = float((p >= schwelle).mean())
    fpr = 1.0 - spez
    return (q - fpr) / (sens - fpr), q


def em_schaetzer(p, pi_quelle, iters=1000, tol=1e-12):
    """Saerens et al. 2002."""
    pi = float(pi_quelle)
    for _ in range(iters):
        w1, w0 = pi / pi_quelle, (1 - pi) / (1 - pi_quelle)
        neu = float((p * w1 / (p * w1 + (1 - p) * w0)).mean())
        if abs(neu - pi) < tol:
            return float(np.clip(neu, 0, 1))
        pi = neu
    return float(np.clip(pi, 0, 1))


def prior_versatz(pi_von, pi_nach):
    return float(np.log(pi_nach / (1 - pi_nach)) - np.log(pi_von / (1 - pi_von)))


# --------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bilder", type=Path,
                    default=Path("data/vinbigdata/vinbigdata/train"))
    ap.add_argument("--csv", type=Path,
                    default=Path("data/vinbigdata/vinbigdata/train.csv"))
    ap.add_argument("--masks", type=Path, default=None,
                    help="224er Lungenmasken. Ohne sie bleibt point_auc_lung "
                         "leer und nur das freie point_auc wird berichtet.")
    ap.add_argument("--out", type=Path, default=Path("predictions_extern_vindr"))
    ap.add_argument("--lesart", choices=["A", "K", "M"], default="A",
                    help="A vorfestgelegt primaer, K vorfestgelegte "
                         "Nebenrechnung, M nachtraeglich (siehe 43_)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 wie beim Kermany-Lauf, der so 5856 Bilder in 201 s "
                         "geschafft hat. Hoeher geht seit der Verschiebung von "
                         "ListDataset auf Modulebene, kostet auf Windows aber "
                         "erst einmal fuenf Torch-Importe.")
    ap.add_argument("--dml-index", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="Rauchtest")
    a = ap.parse_args(argv)

    import torch
    if a.dml_index is not None:
        import torch_directml
        device = torch_directml.device(a.dml_index)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    print("=" * 74)
    print(f"EXTERNE VALIDIERUNG AUF VinDr-CXR, Lesart {a.lesart}")
    print("=" * 74)
    print(f"  Vorfestlegung: erklaerungen/39_extern_vindr_vorfestlegung.md")
    print(f"  Leck (schon gemessen, Lesart A): {LECK:.4f}")

    kal, kurven, schwelle = lade_kalibrierung(
        Path("serving/model/kalibrierung_p10.json"))
    print(f"  Schwelle {schwelle:.6f}, Bezugspraevalenz "
          f"{kal['dev']['praevalenz']:.4f}")

    d, ziel = sammle(a.bilder, a.csv, a.lesart)
    if a.limit:
        d = d.sample(min(a.limit, len(d)), random_state=0).reset_index(drop=True)
        print(f"  RAUCHTEST auf {len(d)} Bildern, das ist KEIN Ergebnis")
    prae = float(d.y.mean())
    print(f"  n {len(d)}, positiv {int(d.y.sum())}, Praevalenz {prae:.4f}")
    print(f"  Priorversatz gegen die Entwicklung: "
          f"{prior_versatz(kal['dev']['praevalenz'], prae):+.4f}")

    print("\n  Gewichte:")
    modelle, pruef = lade_modelle(device)

    print("\n  Vorwaertsrechnen ...")
    p_roh, felder = rechne(d.path.tolist(), baue_kette(a.size), modelle,
                           device, a.batch, a.workers)

    p_kal = np.stack([platt_apply(p_roh[:, k], *kurven[k]) for k in FOLDS], axis=1)
    p_ens = p_kal.mean(axis=1)
    p_ens_roh = p_roh.mean(axis=1)
    y = d.y.to_numpy()

    # ---- A
    auc, lo, hi = boot_auc(y, p_ens)
    auc_roh, _, _ = boot_auc(y, p_ens_roh)
    print("\n" + "=" * 74)
    print("A  erkennt das Modell die Verschattung?")
    print("=" * 74)
    print(f"  roh gepoolt            {auc:.4f}  [{lo:.4f}, {hi:.4f}]")
    print(f"  ohne Platt gemittelt   {auc_roh:.4f}")

    # Die Vorfestlegung nennt die LECK-BEREINIGTE Zahl als die zu berichtende.
    # Die rohe steht daneben und ist nicht das Urteil.
    leck_ist, leck_oof = leck_score(d)
    # Der Vergleichswert gehoert zur LESART, nicht zum Datensatz: ein anderes
    # Ziel-Label heisst eine andere Geometrie-Trennung. Beim ersten M-Lauf
    # schlug die Warnung an, weil sie gegen die 0,7822 der Lesart A hielt.
    # Das war ein Fehlalarm und genau die Sorte, die man abstellt, bevor man
    # sich an sie gewoehnt.
    soll = LECK.get(a.lesart)
    if soll is not None and abs(leck_ist - soll) > 5e-3:
        print(f"  ACHTUNG: Leck hier {leck_ist:.4f}, Vorpruefung sagte "
              f"{soll:.4f} fuer Lesart {a.lesart}. Nicht dieselbe Menge?")
    elif soll is None:
        print(f"  (Lesart {a.lesart} hat keinen vorgemerkten Leck-Wert, "
              f"gerechnet {leck_ist:.4f})")
    a_leck, je_quintil = leck_bereinigt(y, p_ens, leck_oof)
    print(f"\n  Leck nachgerechnet     {leck_ist:.4f}")
    print(f"  {'Q':>2}{'n':>7}{'pos':>6}{'neg':>7}{'Paare':>10}{'AUC':>9}")
    for r in je_quintil:
        wert = f"{r['auc']:.4f}" if r["auc"] is not None else "  zu duenn"
        print(f"  {r['q']:>2}{r['n']:>7}{r['n_pos']:>6}{r['n_neg']:>7}"
              f"{r['paare']:>10}{wert:>9}")
    print(f"\n  A LECK-BEREINIGT       {a_leck:.4f}   <- die Zahl der Vorfestlegung")
    print(f"  intern (Kreuzvalid.)   {INTERN_A:.4f}")
    print(f"  Tor {TOR_A:.2f}: "
          f"{'BESTANDEN' if a_leck >= TOR_A else 'DURCHGEFALLEN'}")

    # ---- Kopffeld, nur auf den Positiven, je Radiologe
    print("\n" + "=" * 74)
    print("KOPFFELD  zeigt es dorthin, wo der Befund ist?")
    print("=" * 74)
    feld_mittel = felder.mean(axis=1)                # ueber die fuenf Modelle
    zeilen, fehlende_maske = [], 0
    for i in np.where(y == 1)[0]:
        r = d.iloc[i]
        heat = feld_auf_referenz(feld_mittel[i])
        lung = load_lung(a.masks, r.image_id) if a.masks else None
        if a.masks and lung is None:
            fehlende_maske += 1
        for rad, bx in kaesten_je_radiologe(ziel, r.image_id,
                                            r.width, r.height).items():
            m = evaluate_map(heat, box_mask(bx, REF_SIZE, BOX_SPACE), lung)
            zeilen.append({"image_id": r.image_id, "rad_id": rad,
                           "n_rad": int(r.n_rad), **m})
    k = pd.DataFrame(zeilen)
    if fehlende_maske:
        print(f"  {fehlende_maske} Bilder ohne Lungenmaske")

    spalte = "point_auc_lung" if (a.masks and k.point_auc_lung.notna().any()) \
        else "point_auc"
    if spalte == "point_auc":
        print("  KEINE Lungenmasken: berichtet wird das freie point_auc, das")
        print("  ist NICHT die Groesse, aus der die internen 0,9123 stammen.")

    m_kopf, m_lo, m_hi = boot_mittel(k.groupby("image_id")[spalte].mean())
    print(f"\n  {spalte}  {m_kopf:.4f}  [{m_lo:.4f}, {m_hi:.4f}]"
          f"   auf {k.image_id.nunique()} Bildern")
    print(f"  intern {INTERN_KOPF:.4f}, Lagepriore 0,7520, Grad-CAM 0,7312")
    print(f"  Tor {TOR_KOPF:.2f}: "
          f"{'BESTANDEN' if m_kopf >= TOR_KOPF else 'DURCHGEFALLEN'}")

    print("\n  je Radiologe (nur die mit mindestens 30 Bildern):")
    je_rad = {}
    for rad, g in k.groupby("rad_id"):
        if len(g) < 30:
            continue
        mm, ml, mh = boot_mittel(g[spalte])
        je_rad[rad] = {"n": int(len(g)), "wert": mm, "lo": ml, "hi": mh}
        print(f"    {rad:<5} n {len(g):>4}   {mm:.4f}  [{ml:.4f}, {mh:.4f}]")
    if je_rad:
        spanne = max(v["wert"] for v in je_rad.values()) - \
            min(v["wert"] for v in je_rad.values())
        print(f"\n  Spanne zwischen den Radiologen: {spanne:.4f}")
        print("  Das ist die Obergrenze dessen, was hier messbar ist.")

    print("\n  nach Zustimmung:")
    nach_zust = {}
    for n_rad, g in k.groupby("n_rad"):
        mm, ml, mh = boot_mittel(g[spalte])
        nach_zust[int(n_rad)] = {"n": int(len(g)), "wert": mm}
        print(f"    {int(n_rad)} Radiologe(n)  n {len(g):>4}   {mm:.4f}"
              f"  [{ml:.4f}, {mh:.4f}]")

    # ---- Kalibrierung
    print("\n" + "=" * 74)
    print("KALIBRIERUNG  traegt die Zahl als Zahl?")
    print("=" * 74)
    e, br = ece(y, p_ens), brier(y, p_ens)
    print(f"  ECE {e:.4f} gegen intern 0,0094")
    print(f"  Brier {br:.4f}")
    versatz = prior_versatz(kal["dev"]["praevalenz"], prae)
    p_korr = 1 / (1 + np.exp(-(logit(p_ens) + versatz)))
    print(f"  nach Priorverschiebung ({versatz:+.4f}): "
          f"ECE {ece(y, p_korr):.4f}, Brier {brier(y, p_korr):.4f}")
    ap_ = arbeitspunkt(y, p_ens, schwelle)
    print(f"\n  bei {schwelle:.4f}:  Sens {ap_['sens']:.4f}  Spez {ap_['spez']:.4f}"
          f"  PPV {ap_['ppv']:.4f}  NPV {ap_['npv']:.4f}")
    print(f"  intern bei derselben Schwelle: Sens "
          f"{kal['dev_bei_schwelle']['sens']:.4f}, Spez "
          f"{kal['dev_bei_schwelle']['spez']:.4f}")

    # ---- Praevalenzschaetzer, zweiter Test nach Kermany (siehe 37_)
    print("\n" + "=" * 74)
    print("PRAEVALENZSCHAETZER  laesst sich die Haeufigkeit aus dem Strom lesen?")
    print("=" * 74)
    s_sens = kal["dev_bei_schwelle"]["sens"]
    s_spez = kal["dev_bei_schwelle"]["spez"]
    roh, q_pos = bbse(p_ens, schwelle, s_sens, s_spez)
    em_wert = em_schaetzer(p_ens, kal["dev"]["praevalenz"])
    print(f"\n  wahre Praevalenz        {prae:.4f}")
    print(f"  als positiv eingestuft  {q_pos:.4f}   "
          f"(Falschalarmrate der Quelle allein: {1 - s_spez:.4f})")
    print(f"  BBSE roh                {roh:+.4f}  -> beschnitten "
          f"{max(roh, 0.0):.4f}  ({max(roh, 0.0) - prae:+.4f})")
    print(f"  EM                      {em_wert:.4f}  ({em_wert - prae:+.4f})")
    if roh < 0:
        print("\n  Der Rohwert ist NEGATIV. Es werden weniger Bilder positiv")
        print("  eingestuft, als die Quelle allein an Falschalarmen erzeugt.")
        print("  Unter Label Shift ist das unmoeglich. Der Schaetzer meldet")
        print("  damit nicht eine Zahl, sondern den Bruch seiner eigenen Annahme.")

    # ---- ablegen
    a.out.mkdir(parents=True, exist_ok=True)
    for j, kf in enumerate(FOLDS):
        d[f"p_roh_f{kf}"] = p_roh[:, j]
        d[f"p_kal_f{kf}"] = p_kal[:, j]
    d["p_ens"] = p_ens
    d["p_ens_roh"] = p_ens_roh
    d.to_csv(a.out / f"extern_vindr_ens_{a.lesart}.csv", index=False)
    k.to_csv(a.out / f"extern_vindr_kopf_{a.lesart}.csv", index=False)

    bericht = {
        "wann": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "vorfestlegung": "erklaerungen/39_extern_vindr_vorfestlegung.md",
        "arm": ARM_TAG, "lesart": a.lesart, "rauchtest": bool(a.limit),
        "n": int(len(d)), "positiv": int(d.y.sum()), "praevalenz": prae,
        "checkpoints": pruef, "schwelle": schwelle,
        "leck_vorher_gemessen": LECK.get(a.lesart),
        "a": {"auc_roh_gepoolt": auc, "lo": lo, "hi": hi,
              "auc_ohne_platt": auc_roh, "leck_nachgerechnet": leck_ist,
              "auc_leck_bereinigt": a_leck, "je_quintil": je_quintil,
              "tor": TOR_A, "bestanden": bool(a_leck >= TOR_A)},
        "praevalenzschaetzer": {"wahr": prae, "pos_rate": q_pos,
                                "bbse_roh": roh, "bbse": max(roh, 0.0),
                                "em": em_wert,
                                "annahme_gebrochen": bool(roh < 0)},
        "kopffeld": {"spalte": spalte, "wert": m_kopf, "lo": m_lo, "hi": m_hi,
                     "n_bilder": int(k.image_id.nunique()),
                     "intern": INTERN_KOPF, "lagepriore": 0.7520,
                     "tor": TOR_KOPF, "bestanden": bool(m_kopf >= TOR_KOPF),
                     "je_radiologe": je_rad, "nach_zustimmung": nach_zust},
        "kalibrierung": {"ece": e, "brier": br, "priorversatz": versatz,
                         "ece_nach_prior": ece(y, p_korr),
                         "brier_nach_prior": brier(y, p_korr),
                         "zuverlaessigkeit": zuverlaessigkeit(p_ens, y)},
        "arbeitspunkt": ap_,
        "c_messbar": False,
        "c_hinweis": "512er-PNG ohne DICOM-Kopfzeilen, keine ViewPosition. "
                     "C bleibt auf RSNA gemessen, siehe erklaerungen/39_.",
        "dauer_s": round(time.time() - t0, 1),
    }
    (a.out / f"bericht_{a.lesart}.json").write_text(
        json.dumps(bericht, indent=2, ensure_ascii=False))

    print("\n" + "=" * 74)
    print("URTEIL")
    print("=" * 74)
    print(f"  primaer  Kopffeld {m_kopf:.4f} gegen Tor {TOR_KOPF:.2f}: "
          f"{'BESTANDEN' if m_kopf >= TOR_KOPF else 'DURCHGEFALLEN'}")
    print(f"  zweitens A leck-bereinigt {a_leck:.4f} gegen Tor {TOR_A:.2f}: "
          f"{'BESTANDEN' if a_leck >= TOR_A else 'DURCHGEFALLEN'}")
    print(f"  C: NICHT GEMESSEN, und das war vorher so festgelegt.")
    print(f"\n  {time.time() - t0:.0f} s  ->  {a.out}")
    if a.limit:
        print("\n  ACHTUNG: Rauchtest. Diese Zahlen sind kein Ergebnis.")


if __name__ == "__main__":
    main()
