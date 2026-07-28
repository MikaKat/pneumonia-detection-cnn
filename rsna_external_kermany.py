"""
Schritt 10: Externe Validierung -- das RSNA-Modell auf Kermany, nur Inferenz.

Warum das der wertvollste noch fehlende Punkt ist
--------------------------------------------------
Alles bisher Gemessene stammt aus EINEM Datensatz. Kreuzvalidierung schuetzt
gegen Ueberanpassung an eine Stichprobe, nicht gegen Ueberanpassung an eine
Aufnahmesituation. Die Frage, die ein Radiologe stellt, ist eine andere:
funktioniert das an einem anderen Haus, an anderem Geraet, an anderen Patienten?

Der Wechsel ist hier ungewoehnlich hart, und das ist Absicht:

  * RSNA: Erwachsene, US-Notaufnahmen, DICOM 1024x1024, viele Liegendaufnahmen,
    Positivrate 0,225.
  * Kermany: Kinder von 1-5 Jahren, Guangzhou, JPEG in wechselnden Groessen,
    Positivrate 0,74.

Alter, Land, Geraet, Dateiformat und Praevalenz aendern sich gleichzeitig. Ein
Modell, das das ueberlebt, hat etwas ueber Lungen gelernt. Ein Einbruch ist
kein Misserfolg, sondern das Ergebnis -- solange die Ursache benannt wird.

NICHTS wird hier trainiert. Es laufen die fuenf vorhandenen RSNA-Checkpoints.

Vier Kontrollen, ohne die die Zahl nichts wert waere
----------------------------------------------------
1. **Der Metadaten-Leak von Kermany.** Allein die JPEG-Abmessungen trennen die
   Klassen mit AUC ~0,91. Beim Skalieren auf 224x224 wird daraus ein
   systematischer Unterschied im Downsampling-Faktor und im Seitenverhaeltnis --
   also ein Kanal, den auch ein fremdes Modell versehentlich anzapfen kann.
   Deshalb wird der Header-Score mitberechnet UND die Modell-AUC INNERHALB
   seiner Quintile berichtet. Genau die Schichtung, die auf RSNA fuer
   ViewPosition noetig war.

2. **Strecken gegen quadratisch fuellen.** `build_transforms` skaliert stur auf
   224x224. RSNA-Bilder sind quadratisch, Kermany-Bilder nicht -- das Modell
   bekaeme also verzerrte Lungen zu sehen, eine Verzerrung, die es nie gelernt
   hat, und die zudem mit der Klasse korreliert (siehe 1). Beide Varianten
   laufen, die Differenz ist selbst ein Messwert.

3. **Gruppierter Bootstrap.** Ein Kermany-Patient hat mehrere Aufnahmen
   (`person17_bacteria_43.jpeg`). Ein Bild-weiser Bootstrap taete so, als
   waeren das unabhaengige Faelle, und lieferte ein zu enges Intervall.
   Gezogen werden Patientengruppen, nicht Bilder.

4. **Schwellen-Uebertragung.** Die auf RSNA gefundene Schwelle wird unveraendert
   angewandt. Bei einer Praevalenz, die von 0,225 auf 0,74 springt, ist das
   Ergebnis vorhersehbar schlecht -- aber genau das passiert, wenn ein Modell
   ohne Rekalibrierung an ein anderes Haus geht. Die Zahl gehoert berichtet,
   nicht versteckt.

CLI:
  python rsna_external_kermany.py --device directml
  python rsna_external_kermany.py --split test        # nur der offizielle Testordner
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

EXTS = ("*.jpeg", "*.jpg", "*.png")
CLASSES = {"NORMAL": 0, "PNEUMONIA": 1}


# --------------------------------------------------------------------------
# Daten einsammeln  (torch-frei)
# --------------------------------------------------------------------------

def collect_kermany(root: Path, splits: list[str] | None = None) -> pd.DataFrame:
    """Alle Kermany-Bilder mit Label, Original-Split und Patientengruppe.

    Die Gruppenlogik kommt aus `splits.parse_record` -- nicht nachgebaut,
    sondern wiederverwendet, damit die Patientengruppen exakt dieselben sind
    wie ueberall sonst im Projekt.
    """
    from splits import parse_record

    rows = []
    for f in sorted(x for x in Path(root).rglob("*")
                    if x.suffix.lower() in {".jpeg", ".jpg", ".png"}):
        rec = parse_record(f.relative_to(root))
        if rec is None:
            continue
        if splits and rec["split"] not in splits:
            continue
        rec["path"] = str(f)
        rows.append(rec)
    return pd.DataFrame(rows)


def read_dims(paths: list[str]) -> pd.DataFrame:
    """Nur die Bildabmessungen -- PIL dekodiert dafuer nicht, das geht in Sekunden."""
    w, h = [], []
    for p in paths:
        with Image.open(p) as im:
            w.append(im.size[0])
            h.append(im.size[1])
    w, h = np.array(w, float), np.array(h, float)
    return pd.DataFrame({"width": w, "height": h, "aspect": w / h,
                         "pixels": w * h})


def grouped_bootstrap_auc(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                          B: int = 500, seed: int = 0) -> tuple[float, float, float]:
    """AUC mit Konfidenzintervall aus gezogenen PATIENTENGRUPPEN.

    Ein bildweiser Bootstrap unterstellt Unabhaengigkeit, die es nicht gibt:
    mehrere Aufnahmen desselben Kindes sind fast dasselbe Bild. Das Intervall
    waere dann zu eng und die externe Zahl schiene sicherer, als sie ist.
    """
    from sklearn.metrics import roc_auc_score

    point = float(roc_auc_score(y, p))
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        if len(set(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    if not out:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def header_leak(dims: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                seed: int = 0) -> tuple[float, np.ndarray]:
    """Wie gut trennen die reinen Abmessungen? Gruppiert kreuzvalidiert.

    Gibt AUC und den Out-of-Fold-Score zurueck. Der Score wird gebraucht, um
    die Modell-AUC INNERHALB seiner Quintile zu berichten -- die Schichtung,
    die feststellt, ob das Modell mehr kann als der Leak.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    X = dims[["width", "height", "aspect", "pixels"]].values
    oof = np.zeros(len(y))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups):
        m = GradientBoostingClassifier(random_state=seed).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof)), oof


def stratified_by_score(y: np.ndarray, p: np.ndarray, strat: np.ndarray,
                        q: int = 5) -> tuple[float, list]:
    """Modell-AUC innerhalb der Quintile eines Stoerscores.

    Liegt sie deutlich ueber 0,5, sieht das Modell etwas, das der Leak nicht
    schon liefert. Faellt sie auf 0,5, war die Gesamtzahl der Leak.

    GEWICHTET NACH DISKORDANTEN PAAREN (n_pos * n_neg), NICHT nach n.
    Das ist keine Feinheit, sondern der Unterschied zwischen einer belastbaren
    und einer irrefuehrenden Zahl: Wenn der Stoerscore stark ist -- auf Kermany
    trennt er mit AUC 0,916 --, dann sind die oberen Quintile fast reinrassig
    positiv. Gemessen wurde hier ein Quintil mit 1182 positiven und **einem
    einzigen** negativen Bild. Seine AUC beruht auf 1182 Paaren, die alle
    dasselbe eine Bild enthalten -- praktisch Rauschen. Nach n gewichtet zieht
    dieses Rauschen mit vollem Gewicht ins Mittel (0,853); nach diskordanten
    Paaren gewichtet, also nach der tatsaechlich vorhandenen Information,
    ergibt sich 0,885.

    Die erste Fassung gewichtete nach n. Das war falsch.
    """
    from sklearn.metrics import roc_auc_score

    edges = np.quantile(strat, np.linspace(0, 1, q + 1))
    edges[-1] += 1e-9
    per, num, den = [], 0.0, 0
    for i in range(q):
        sel = (strat >= edges[i]) & (strat < edges[i + 1])
        npos = int(y[sel].sum())
        nneg = int(sel.sum()) - npos
        pairs = npos * nneg
        if pairs == 0 or sel.sum() < 30:
            per.append((i + 1, int(sel.sum()), npos, nneg, pairs,
                        float("nan")))
            continue
        a = float(roc_auc_score(y[sel], p[sel]))
        per.append((i + 1, int(sel.sum()), npos, nneg, pairs, a))
        num += a * pairs
        den += pairs
    return (num / den if den else float("nan")), per


def operating_point(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    tp = int(((p >= thr) & (y == 1)).sum())
    fp = int(((p >= thr) & (y == 0)).sum())
    fn = int(((p < thr) & (y == 1)).sum())
    tn = int(((p < thr) & (y == 0)).sum())
    return {"sens": tp / max(tp + fn, 1), "spec": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1),
            "pos_rate": float((p >= thr).mean())}


# --------------------------------------------------------------------------
# Torch-Teil
# --------------------------------------------------------------------------

class PadToSquare:
    """Auf ein Quadrat auffuellen statt zu strecken -- mit dem Bildmedian.

    Schwarze Balken waeren selbst ein auffaelliges Merkmal und wuerden eine
    neue Kante einfuehren; derselbe Grund, aus dem `MaskCorners` den Median
    nimmt und nicht Null.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        s = max(w, h)
        med = int(np.median(np.asarray(img.convert("L"))))
        out = Image.new(img.mode, (s, s), med)
        out.paste(img, ((s - w) // 2, (s - h) // 2))
        return out


def build_variants(size: int):
    """Die zwei Vorverarbeitungen, zwischen denen entschieden werden muss."""
    import torchvision.transforms as T

    from rsna_train import IMNET_MEAN, IMNET_STD

    base = [T.Grayscale(num_output_channels=3), T.ToTensor(),
            T.Normalize(IMNET_MEAN, IMNET_STD)]
    return {
        # genau das, was rsna_train.build_transforms(size, False) tut
        "stretch": T.Compose([T.Resize((size, size))] + base),
        # geometrieerhaltend -- so sieht ein Kermany-Bild dem RSNA-Modell
        # aehnlicher, weil RSNA-Bilder quadratisch sind
        "pad": T.Compose([PadToSquare(), T.Resize((size, size))] + base),
    }


class ListDataset:
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("L"))


def predict_all(paths, tf, models, device, batch: int, workers: int):
    """Ein Durchlauf ueber die Daten, alle fuenf Modelle je Batch.

    Fuenf getrennte Durchlaeufe waeren fuenfmal JPEG-Dekodierung -- und die
    ist hier teurer als der Forward-Pass eines ResNet18.
    """
    import torch
    from torch.utils.data import DataLoader

    dl = DataLoader(ListDataset(list(paths), tf), batch_size=batch,
                    num_workers=workers)
    out = [[] for _ in models]
    with torch.no_grad():
        for bi, x in enumerate(dl, 1):
            x = x.to(device)
            for k, m in enumerate(models):
                out[k].append(torch.sigmoid(m(x).squeeze(1)).float().cpu().numpy())
            if bi % 20 == 0:
                print(f"      Batch {bi}/{len(dl)}")
    return np.stack([np.concatenate(o) for o in out], axis=1)     # [n, n_folds]


def load_state_cpu(path: Path):
    """Checkpoint IMMER auf die CPU laden, nie direkt aufs Zielgeraet.

    `torch.load(..., map_location=<DirectML-Geraet>)` stirbt: Torch reicht das
    torch.device-Objekt an `torch_directml.device()` weiter, und die Funktion
    erwartet dort einen Integer-Index. Der Fehler lautet dann

        TypeError: '>=' not supported between instances of 'torch.device' and 'int'

    und sieht aus, als laege es am Checkpoint. Der richtige Weg ist ohnehin
    dieser: state_dict auf die CPU, Modell aufs Geraet, dann `load_state_dict`
    kopiert die Gewichte hinueber.

    `weights_only=True` unterdrueckt zugleich die FutureWarning -- der
    Checkpoint ist ein reines state_dict, mehr wird nicht gebraucht. Der
    Fallback faengt nur alte Torch-Versionen ohne dieses Argument ab.
    """
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def load_models(folds, seed, device):
    from rsna_train import make_model

    models = []
    for f in folds:
        ck = Path(f"checkpoints/rsna_f{f}_s{seed}.pth")
        if not ck.exists():
            print(f"  Checkpoint fehlt, uebersprungen: {ck}")
            continue
        m = make_model(device)                      # legt das Modell aufs Geraet
        m.load_state_dict(load_state_cpu(ck))       # kopiert CPU -> Geraet
        m.eval()
        models.append(m)
    return models


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, default=Path("data/chest_xray"))
    p.add_argument("--split", nargs="*", default=None,
                   help="nur diese Original-Splits (train/val/test); leer = alle")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--thr", type=float, default=None,
                   help="RSNA-Schwelle; Standard = Mittel aus results_rsna.csv")
    p.add_argument("--out", type=Path,
                   default=Path("predictions_rsna/external_kermany.csv"))
    args = p.parse_args(argv)

    from rsna_train import pick_device

    d = collect_kermany(args.images, args.split)
    if d.empty:
        print(f"FEHLER: keine Bilder unter {args.images}")
        return 2
    y = d.label.values
    groups = d.group.values
    print(f"Kermany: {len(d)} Bilder, {d.group.nunique()} Patientengruppen | "
          f"Positivrate {y.mean():.3f}")
    print(f"  je Original-Split: {dict(d.split.value_counts())}")
    print("  (RSNA-Positivrate war 0.225 -- die Praevalenz kippt um Faktor 3)")

    # ---- Kontrolle 1: der Metadaten-Leak, VOR jeder Modellzahl -------------
    dims = read_dims(d.path.tolist())
    leak_auc, leak_score = header_leak(dims, y, groups, args.seed)
    print(f"\nMetadaten-Leak (nur Abmessungen, gruppiert CV): AUC {leak_auc:.3f}")
    print(f"  Seitenverhaeltnis allein: "
          f"{max(np.corrcoef(dims.aspect, y)[0, 1], -1):+.3f} Korrelation zur Klasse")
    print("  Jede Modellzahl unten muss gegen diese Schwelle gelesen werden.")

    device, _ = pick_device(args.device)
    print(f"\nGeraet: {device}")
    models = load_models(args.folds, args.seed, device)
    if not models:
        print("FEHLER: kein Checkpoint gefunden.")
        return 2
    print(f"  {len(models)} Checkpoints geladen")

    thr = args.thr
    if thr is None:
        try:
            thr = float(pd.read_csv("results_rsna.csv")["thr"].mean())
        except Exception:
            thr = 0.5
    print(f"  uebertragene RSNA-Schwelle: {thr:.4f}")

    results = {}
    for name, tf in build_variants(args.size).items():
        print(f"\n  Variante '{name}' ...")
        P = predict_all(d.path.tolist(), tf, models, device, args.batch,
                        args.workers)
        results[name] = P
        for k in range(P.shape[1]):
            d[f"p_{name}_f{args.folds[k]}"] = P[:, k]
        d[f"p_{name}_ens"] = P.mean(axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    d.assign(**{c: dims[c] for c in dims.columns}).to_csv(args.out, index=False)

    # ---- Bericht ----------------------------------------------------------
    print("\n" + "=" * 76)
    print("EXTERNE AUC -- RSNA-Modell, Kermany-Daten, nur Inferenz")
    print("=" * 76)
    print(f"  Referenz intern (RSNA, geschichtet): 0.845 +- 0.015")
    print(f"  Referenz Metadaten-Leak hier:        {leak_auc:.3f}")
    print()
    from sklearn.metrics import roc_auc_score

    ens_auc = {}
    for name, P in results.items():
        per = [float(roc_auc_score(y, P[:, k])) for k in range(P.shape[1])]
        ens = P.mean(axis=1)
        a, lo, hi = grouped_bootstrap_auc(y, ens, groups, seed=args.seed)
        ens_auc[name] = a
        print(f"  {name:<8} je Fold {np.mean(per):.3f} +- {np.std(per, ddof=1):.3f}"
              f"   Ensemble {a:.3f} [{lo:.3f} - {hi:.3f}]  (gruppierter Bootstrap)")
        print(f"           Einzelfolds: {', '.join(f'{v:.3f}' for v in per)}")

    if len(ens_auc) > 1:
        diff = ens_auc["pad"] - ens_auc["stretch"]
        print(f"\n  'pad' minus 'stretch': {diff:+.3f}")
        print("  Positiv heisst: das Modell leidet unter der Verzerrung, die")
        print("  build_transforms bei nicht-quadratischen Bildern erzeugt --")
        print("  ein Vorverarbeitungsfehler, kein Domaenenproblem.")

    best = max(ens_auc, key=ens_auc.get)
    ens = results[best].mean(axis=1)

    print("\n" + "-" * 76)
    print(f"GESCHICHTET nach dem Metadaten-Leak (Variante '{best}')")
    print("-" * 76)
    s_auc, per = stratified_by_score(y, ens, leak_score)
    print(f"  {'Q':<3}{'n':>7}{'n_pos':>8}{'n_neg':>8}{'disk.Paare':>13}{'AUC':>9}")
    thin = []
    for q, n, npos, nneg, pairs, a in per:
        txt = f"{a:.3f}" if not np.isnan(a) else "   --"
        print(f"  {q:<3}{n:>7}{npos:>8}{nneg:>8}{pairs:>13}{txt:>9}")
        if min(npos, nneg) < 30:
            thin.append(q)
    print(f"\n  gewichtetes Mittel (nach diskordanten Paaren): {s_auc:.3f}")
    print(f"  roh, ohne Schichtung:                         "
          f"{float(roc_auc_score(y, ens)):.3f}")
    print(f"  Metadaten-Leak allein:                        {leak_auc:.3f}")
    if thin:
        print(f"\n  Hinweis: Quintil(e) {thin} sind fast reinrassig (unter 30 Bilder")
        print("  der Minderheitsklasse). Ihre AUC traegt kaum Information -- genau")
        print("  deshalb wird nach diskordanten Paaren gewichtet und nicht nach n.")
        good = [q for q, n, npos, nneg, pr, a in per if min(npos, nneg) >= 30
                and not np.isnan(a)]
        sel = np.zeros(len(y), bool)
        edges = np.quantile(leak_score, np.linspace(0, 1, 6))
        edges[-1] += 1e-9
        for q in good:
            sel |= (leak_score >= edges[q - 1]) & (leak_score < edges[q])
        if sel.sum() > 50 and len(set(y[sel])) > 1:
            print(f"  Nur die gut besetzten Quintile {good}: n={int(sel.sum())}, "
                  f"AUC {float(roc_auc_score(y[sel], ens[sel])):.3f}")
    print("\n  Bleibt das deutlich ueber 0,5, sieht das Modell mehr als den Leak.")

    print("\n" + "-" * 76)
    print(f"SCHWELLEN-UEBERTRAGUNG ohne Rekalibrierung (Schwelle {thr:.3f})")
    print("-" * 76)
    op = operating_point(y, ens, thr)
    print(f"  Sensitivitaet {op['sens']:.3f} | Spezifitaet {op['spec']:.3f} | "
          f"PPV {op['ppv']:.3f} | NPV {op['npv']:.3f}")
    print(f"  Anteil positiv vorhergesagt {op['pos_rate']:.3f} "
          f"gegen tatsaechlich {y.mean():.3f}")
    print("  Die Praevalenz springt von 0,225 auf "
          f"{y.mean():.3f}. Eine uebertragene Schwelle KANN hier nicht passen --")
    print("  die Zahl zeigt, was ein Einsatz ohne Rekalibrierung bedeutet.")

    print("\n" + "=" * 76)
    print(f"Rohdaten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
