"""
Vergleicht crop / softmask / hardmask unter identischen Bedingungen.

Ohne externes Testset ist ein einzelner Lauf pro Variante nicht aussagekraeftig
-- Unterschiede von 1-2 Prozentpunkten AUC sind reines Rauschen. Deshalb:

  * dieselben patientenweisen Folds fuer alle Varianten (aus splits.json)
  * mehrere Seeds pro Fold
  * gepaarte Auswertung (Variante A vs B auf demselben Fold/Seed)
  * Robustheitstest: kuenstliche Stoerungen auf dem Val-Set. Eine Variante, die
    dabei weniger einbricht, stuetzt sich weniger auf Aufnahme-Artefakte --
    das ist der beste Ersatz fuer ein externes Testset, den du hier hast.
  * Checkpoint-Auswahl auf einem INNEREN Split aus fold["train"]. Das aeussere
    Val-Set wird ausschliesslich berichtet, nie optimiert.
  * Alle Val-Vorhersagen landen als CSV auf der Platte, damit jede
    Nachanalyse ohne Retraining moeglich ist.

CLI:
  python train_compare.py --prepared data/prepared --splits splits.json \
      --variants crop softmask hardmask --folds 0 1 2 3 4 --seeds 0 1 --epochs 12
"""

from __future__ import annotations

import argparse
import json
import random
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, resnet18

from splits import parse_record  # dieselbe Patienten-Gruppierung wie im Split-Skript

IMNET_MEAN, IMNET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------
# Daten
# --------------------------------------------------------------------------

class XrayDataset(Dataset):
    def __init__(self, root: Path, files: list[str], labels: dict[str, int], tf):
        self.root, self.files, self.labels, self.tf = root, files, labels, tf

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        img = Image.open(self.root / f).convert("L")
        return self.tf(img), float(self.labels[f])


class RandomResolution:
    """Zufaelliges Herunter- und Wiederhochskalieren (Aufloesungs-Jitter).

    Warum das noetig ist: NORMAL-Aufnahmen sind in diesem Datensatz gut doppelt
    so gross wie PNEUMONIA-Aufnahmen und werden beim Crop staerker verkleinert
    (Zoomfaktor-AUC 0.897). Die absolute Bildschaerfe verraet damit das Label,
    ohne dass Anatomie im Spiel waere.

    Der Jitter loescht diesen Kanal, ohne echtes Signal zu zerstoeren: Die
    Konsolidierung selbst aeussert sich als RAEUMLICHES Muster -- homogene
    Verschattung mit fehlender Gefaesszeichnung an einer bestimmten Stelle --
    und das ueberlebt eine zufaellige Skalierung. Was nicht ueberlebt, ist ein
    global kalibrierter Schaerfewert, und genau der soll weg.

    Deshalb bewusst deterministisch NICHT angleichen: die auf crop_side
    gematchte Schaerfe-AUC von 0.744 ist ueberwiegend echter Befund
    (Konsolidierung loescht feine Zeichnung). Global weichzeichnen wuerde ihn
    mit entfernen.
    """

    def __init__(self, size: int, lo: float = 0.35, hi: float = 1.0, p: float = 0.8):
        self.size, self.lo, self.hi, self.p = size, lo, hi, p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        f = random.uniform(self.lo, self.hi)
        s = max(32, int(round(self.size * f)))
        return (img.resize((s, s), Image.BILINEAR)
                   .resize((self.size, self.size), Image.BILINEAR))


def build_transforms(size: int, train: bool, res_jitter: bool = True):
    # KEIN horizontales Spiegeln: das erzeugt Situs inversus, vertauscht die
    # Herzsilhouette und widerspricht dem R-Marker im Bild.
    base = [T.Grayscale(num_output_channels=3), T.ToTensor(),
            T.Normalize(IMNET_MEAN, IMNET_STD)]
    if not train:
        return T.Compose([T.Resize((size, size))] + base)
    aug = [T.Resize((size, size))]
    if res_jitter:
        aug.append(RandomResolution(size))
    aug += [
        T.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.93, 1.07)),
        T.ColorJitter(brightness=0.15, contrast=0.15),
    ]
    return T.Compose(aug + base)


PERTURBATIONS = {
    "clean":      lambda s: [T.Resize((s, s))],
    "zoom_in":    lambda s: [T.Resize((int(s * 1.15),) * 2), T.CenterCrop(s)],
    "shift":      lambda s: [T.Resize((s, s)), T.RandomAffine(0, translate=(0.08, 0.08))],
    "rotate":     lambda s: [T.Resize((s, s)), T.RandomRotation(12)],
    "low_contr":  lambda s: [T.Resize((s, s)), T.ColorJitter(contrast=(0.6, 0.6))],
    "bright":     lambda s: [T.Resize((s, s)), T.ColorJitter(brightness=(1.35, 1.35))],
    "blur":       lambda s: [T.Resize((s, s)), T.GaussianBlur(5, sigma=1.6)],
    # gezielte Probe auf den Schaerfe-Shortcut: Aufloesung halbieren und
    # wieder hochziehen. Ein Modell, das absolute Schaerfe als Merkmal nutzt,
    # bricht hier deutlich ein.
    "lowres":     lambda s: [T.Resize((int(s * 0.45),) * 2), T.Resize((s, s))],
}


def perturbed_transform(size: int, name: str):
    return T.Compose(PERTURBATIONS[name](size) +
                     [T.Grayscale(num_output_channels=3), T.ToTensor(),
                      T.Normalize(IMNET_MEAN, IMNET_STD)])


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def pick_device(name: str):
    """Waehlt das Rechengeraet. Gibt (device, pin_memory) zurueck.

    DirectML ist auf dieser Hardware der einzige GPU-Weg: die RX 5500 XT ist
    RDNA1 (gfx1012) und damit weder von ROCm noch von AMDs PyTorch-Preview
    fuer Windows abgedeckt. DirectML laeuft ueber DirectX 12 und nimmt jede
    DX12-faehige Karte -- dafuer ohne AMP und mit gelegentlichem CPU-Fallback
    einzelner Operationen.
    """
    if name in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda"), True
    if name in ("auto", "directml"):
        try:
            import torch_directml
            if torch_directml.is_available():
                return torch_directml.device(), False
        except ImportError:
            if name == "directml":
                raise SystemExit("torch-directml fehlt:  pip install torch-directml")
    if name == "directml":
        raise SystemExit("torch-directml findet kein Geraet.")
    return torch.device("cpu"), False


def make_model(device):
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m.to(device)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    p, y = [], []
    for x, t in loader:
        logit = model(x.to(device, non_blocking=True)).squeeze(1)
        p.append(torch.sigmoid(logit).float().cpu().numpy())
        y.append(t.numpy())
    return np.concatenate(p), np.concatenate(y)


def load_crop_side(prepared: Path) -> dict[str, float]:
    """Zoomfaktor-Protokoll aus lung_preprocess.py, Schluessel ohne Endung."""
    path = prepared / "crop_log.csv"
    if not path.exists():
        print(f"  Hinweis: {path} fehlt -- auc_matched wird nicht berechnet.")
        return {}
    df = pd.read_csv(path)
    return {str(Path(f).with_suffix("")).replace("\\", "/"): float(c)
            for f, c in zip(df["file"], df["crop_side"])}


def caliper_match(y: np.ndarray, x: np.ndarray, caliper: float) -> np.ndarray:
    """1:1-Matching der Minderheitsklasse auf die Mehrheitsklasse, ohne Zuruecklegen.

    Warum das die Vorgaengerversion (5 Quantil-Baender) ersetzt: Die Baender waren
    zu grob. Zwei der fuenf hatten eine Positivrate von 98-99 % und flogen wegen
    min_per_class heraus -- also genau die 40 % der Bilder, bei denen die Groesse
    das Label am deutlichsten verraet. In den drei uebrigen Baendern trennte
    crop_side allein immer noch mit AUC 0.55-0.67. Die Zahl hiess "gematcht",
    war es aber nicht.

    Hier wird jedes Bild der Minderheitsklasse mit dem naechstgelegenen noch
    freien Partner der anderen Klasse gepaart, sofern deren Abstand in
    log(crop_side) unter dem Caliper liegt. Ergebnis: eine 50/50-Kohorte, in der
    crop_side per Konstruktion nichts mehr verraet. Die Selbstkontrolle dazu ist
    `matched_resid` -- liegt sie nicht bei ~0.50, ist die Kohorte kaputt und
    `auc_matched` nicht interpretierbar.

    Das Matching ist deterministisch und haengt nur vom Fold ab, nicht vom
    Modell. Alle Varianten werden damit auf DERSELBEN Kohorte verglichen, die
    gepaarten Differenzen bleiben also gueltig.

    Rueckgabe: Indexarray der Kohorte. Die erste Haelfte ist die
    Minderheitsklasse, die zweite Haelfte sind ihre Partner in derselben
    Reihenfolge -- idx[i] und idx[len(idx)//2 + i] sind also ein Paar.
    """
    y = np.asarray(y)
    x = np.asarray(x, dtype=float)
    a = np.where(y == 0)[0]
    b = np.where(y == 1)[0]
    if len(a) > len(b):            # a = Minderheitsklasse
        a, b = b, a
    order = np.argsort(x[b])
    pool, xs = b[order], x[b][order]
    used = np.zeros(len(pool), dtype=bool)
    keep_a, keep_b = [], []
    for i in a[np.argsort(x[a])]:
        j = int(np.searchsorted(xs, x[i]))
        lo_k, hi_k, best = j - 1, j, -1
        # nach aussen laufen, aufsteigend im Abstand -- der erste freie Treffer
        # ist damit automatisch der naechste
        while lo_k >= 0 or hi_k < len(pool):
            d_lo = x[i] - xs[lo_k] if lo_k >= 0 else np.inf
            d_hi = xs[hi_k] - x[i] if hi_k < len(pool) else np.inf
            if min(d_lo, d_hi) > caliper:
                break
            if d_lo <= d_hi:
                if not used[lo_k]:
                    best = lo_k
                    break
                lo_k -= 1
            else:
                if not used[hi_k]:
                    best = hi_k
                    break
                hi_k += 1
        if best >= 0:
            used[best] = True
            keep_a.append(i)
            keep_b.append(pool[best])
    return np.asarray(keep_a + keep_b, dtype=int)


def matched_scores(y: np.ndarray, p: np.ndarray, cs: np.ndarray,
                   caliper: float, n_boot: int = 2000, seed: int = 0) -> dict:
    """AUC auf der gematchten Kohorte, mit Bootstrap-CI und Selbstkontrolle.

    Das CI ist hier kein Zierrat: die Kohorte umfasst nur ~1/3 des Val-Sets, die
    Unsicherheit ist entsprechend groesser als bei der rohen AUC. Wer die
    gematchte Zahl ohne CI mit der rohen vergleicht, unterschaetzt das Rauschen.
    """
    nan = {"auc_matched": float("nan"), "matched_n": 0,
           "matched_resid": float("nan"),
           "auc_matched_lo": float("nan"), "auc_matched_hi": float("nan")}
    if len(cs) == 0 or np.isnan(cs).any() or (cs <= 0).any():
        return nan
    idx = caliper_match(y, np.log(cs), caliper)
    if len(idx) == 0:
        return nan
    yy, pp, vv = y[idx], p[idx], np.log(cs[idx])
    if min((yy == 0).sum(), (yy == 1).sum()) < 10:
        return nan
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        s = rng.integers(0, len(yy), len(yy))
        if len(np.unique(yy[s])) == 2:
            boot.append(roc_auc_score(yy[s], pp[s]))
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan))
    return {"auc_matched": float(roc_auc_score(yy, pp)),
            "matched_n": int(len(idx)),
            # muss ~0.50 sein, sonst ist der Confounder nicht neutralisiert
            "matched_resid": float(roc_auc_score(yy, -vv)),
            "auc_matched_lo": float(lo), "auc_matched_hi": float(hi)}


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    tpr = tp / max(ys.sum(), 1); fpr = fp / max((1 - ys).sum(), 1)
    return float(ps[int(np.argmax(tpr - fpr))])


def scores(y: np.ndarray, p: np.ndarray, thr: float | None = None) -> dict:
    """Rangmetriken plus Sens/Spez an einer VORGEGEBENEN Schwelle.

    Warum die Schwelle von aussen kommt: Youden-J auf demselben Satz zu suchen,
    auf dem man Sens/Spez berichtet, ist dieselbe Art Zirkelschluss wie die
    Checkpoint-Wahl auf dem Berichtsatz. `sens`/`spec` gelten hier fuer eine auf
    dem Selektions-Split festgelegte Schwelle; `sens_oracle`/`spec_oracle` sind
    das optimistische Gegenstueck, absichtlich mitgeschrieben, damit die Luecke
    zwischen beiden sichtbar bleibt.
    """
    out = {"auc": roc_auc_score(y, p), "auprc": average_precision_score(y, p)}
    t_or = youden_threshold(y, p)
    out["sens_oracle"] = float(((p >= t_or) & (y == 1)).sum() / max((y == 1).sum(), 1))
    out["spec_oracle"] = float(((p < t_or) & (y == 0)).sum() / max((y == 0).sum(), 1))
    t = t_or if thr is None else thr
    out["thr"] = float(t)
    out["sens"] = float(((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1))
    out["spec"] = float(((p < t) & (y == 0)).sum() / max((y == 0).sum(), 1))
    return out


def inner_split(files: list[str], labels, seed: int, n_splits: int
                ) -> tuple[list[str], list[str]]:
    """Teilt fold["train"] patientenweise in Trainings- und Selektionsteil.

    Warum: vorher wurde das Checkpoint per AUC auf dem AEUSSEREN Val-Set gewaehlt
    und dieselbe AUC dann berichtet. Damit ist jede gemeldete Zahl ein Maximum
    ueber alle Epochen auf den Berichtsdaten -- systematisch zu optimistisch, und
    zwar um so mehr, je unruhiger die Epochenkurve einer Variante ist. Auf
    Kermany versteckt die Decke das; bei AUC ~0.87 auf RSNA verschiebt es die
    Zahl um die Groessenordnung der Effekte, die man messen will.

    Gruppiert wird mit derselben Funktion wie in splits.py, sonst leakt der
    Patient hier wieder herein.
    """
    y = np.array([labels[f] for f in files])
    # splits.json wurde unter Windows geschrieben und enthaelt Backslashes.
    # Ohne Normalisierung ist der ganze Pfad unter Linux ein einziges Path-Teil,
    # parse_record findet die Klasse nicht und gibt None zurueck.
    g = np.array([parse_record(Path(f.replace("\\", "/")))["group"] for f in files])
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_i, sel_i = next(iter(sgkf.split(np.zeros(len(files)), y, g)))
    assert not (set(g[tr_i]) & set(g[sel_i])), "Gruppen-Leak im inneren Split!"
    return [files[i] for i in tr_i], [files[i] for i in sel_i]


def run_one(root: Path, labels, fold, fold_k, seed, args, device, crop_side) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    fit_files, sel_files = inner_split(fold["train"], labels, seed, args.inner_splits)

    tr_ds = XrayDataset(root, fit_files, labels,
                        build_transforms(args.size, True, not args.no_res_jitter))
    sel_ds = XrayDataset(root, sel_files, labels, build_transforms(args.size, False))
    va_ds = XrayDataset(root, fold["val"], labels, build_transforms(args.size, False))
    tr = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=args.pin, drop_last=True)
    sel = DataLoader(sel_ds, batch_size=args.batch * 2, num_workers=args.workers)
    va = DataLoader(va_ds, batch_size=args.batch * 2, num_workers=args.workers)

    model = make_model(device)
    y_tr = np.array([labels[f] for f in fit_files])
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                              dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(len(tr), 1))

    best_sel, best_state, best_ep = -1.0, None, -1
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        for x, t in tr:
            x, t = x.to(device, non_blocking=True), t.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            crit(model(x).squeeze(1), t).backward()
            opt.step()
            sched.step()
        ps, ys = predict(model, sel, device)
        auc_sel = roc_auc_score(ys, ps)
        if auc_sel > best_sel:
            best_sel, best_ep = auc_sel, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        dt = time.time() - t0
        eta = dt * (args.epochs - ep - 1)
        print(f"      epoch {ep + 1}/{args.epochs}  sel AUC {auc_sel:.4f}  "
              f"[{dt:.0f}s, Rest ~{eta / 60:.0f} min]")

    # letzte Epoche auf dem aeusseren Val -- Referenz, um zu sehen, wie viel die
    # Checkpoint-Auswahl ueberhaupt bewegt
    p_last, y = predict(model, va, device)
    auc_last = roc_auc_score(y, p_last)

    model.load_state_dict(best_state)
    # Schwelle auf dem Selektions-Split festlegen, nicht auf dem Berichtsatz
    p_sel, y_sel = predict(model, sel, device)
    thr = youden_threshold(y_sel, p_sel)

    p, y = predict(model, va, device)
    res = scores(y, p, thr)
    res.update({"auc_last": float(auc_last), "auc_sel": float(best_sel),
                "best_epoch": best_ep + 1,
                "n_fit": len(fit_files), "n_sel": len(sel_files),
                "n_val": len(fold["val"])})

    # Der ehrliche Wert: AUC auf der crop_side-gematchten Kohorte
    cs = np.array([crop_side.get(str(Path(f).with_suffix("")).replace("\\", "/"), np.nan)
                   for f in fold["val"]])
    res.update(matched_scores(y, p, cs, args.match_caliper, seed=seed)
               if crop_side else matched_scores(y, p, np.array([]), args.match_caliper))
    warn = ("" if res["matched_n"] == 0 or abs(res["matched_resid"] - 0.5) < 0.02
            else "  <-- Kohorte pruefen!")
    print(f"      AUC roh {res['auc']:.4f} (letzte Epoche {auc_last:.4f}) | "
          f"gematcht {res['auc_matched']:.4f} "
          f"[{res['auc_matched_lo']:.4f}-{res['auc_matched_hi']:.4f}], "
          f"n={res['matched_n']}, Restleak {res['matched_resid']:.3f}{warn}")

    # Robustheit: gleiches Modell, gestoerte Val-Bilder
    preds = {"file": list(fold["val"]), "y": y.tolist(), "p_clean": p.tolist(),
             "p_last_epoch": p_last.tolist()}
    names = [n for n in PERTURBATIONS if n != "clean"]
    for i, name in enumerate(names, 1):
        t0 = time.time()
        torch.manual_seed(seed)          # shift/rotate sind stochastisch
        random.seed(seed)
        ds = XrayDataset(root, fold["val"], labels, perturbed_transform(args.size, name))
        pp, yy = predict(model, DataLoader(ds, batch_size=args.batch * 2,
                                           num_workers=args.workers), device)
        res[f"auc_{name}"] = roc_auc_score(yy, pp)
        preds[f"p_{name}"] = pp.tolist()
        print(f"      Robustheit {i}/{len(names)}  {name:<10} "
              f"AUC {res[f'auc_{name}']:.4f}  [{time.time() - t0:.0f}s]")

    pert = [res[f"auc_{n}"] for n in PERTURBATIONS if n != "clean"]
    res["auc_perturbed_mean"] = float(np.mean(pert))
    res["robustness_drop"] = res["auc"] - res["auc_perturbed_mean"]

    # Alles auf Platte: jede Nachanalyse (anderer Caliper, andere Matching-
    # Variable, Kalibrierung, Fehleranalyse) kostet damit Sekunden statt eines
    # neuen 45-Minuten-Laufs.
    if args.pred_dir is not None:
        args.pred_dir.mkdir(parents=True, exist_ok=True)
        out = args.pred_dir / f"{root.name}_f{fold_k}_s{seed}.csv"
        pd.DataFrame(preds).to_csv(out, index=False)
        res["pred_file"] = out.name
    return res


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True, type=Path)
    p.add_argument("--splits", required=True, type=Path)
    p.add_argument("--variants", nargs="+",
                   default=["crop", "softmask", "hardmask", "centercrop"])
    p.add_argument("--no-res-jitter", action="store_true",
                   help="Aufloesungs-Jitter abschalten (Ablation)")
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "directml", "cpu"])
    p.add_argument("--out", type=Path, default=Path("results.csv"))
    p.add_argument("--pred-dir", type=Path, default=Path("predictions"),
                   help="Ordner fuer die Val-Vorhersagen pro Lauf; leer = nicht speichern")
    p.add_argument("--match-caliper", type=float, default=0.01,
                   help="max. Abstand in log(crop_side) beim 1:1-Matching. "
                        "0.01 empirisch kalibriert: Restleak 0.495, ~1/3 des Val-Sets")
    p.add_argument("--inner-splits", type=int, default=6,
                   help="Selektions-Split = 1/N von fold[train] (6 -> ~17 %)")
    args = p.parse_args()
    if args.pred_dir is not None and str(args.pred_dir) in ("", "."):
        args.pred_dir = None

    device, args.pin = pick_device(args.device)
    n_runs = len(args.variants) * len(args.folds) * len(args.seeds)
    print(f"Device: {device} | geplante Laeufe: {n_runs} "
          f"({len(args.variants)} Varianten x {len(args.folds)} Folds x {len(args.seeds)} Seeds)")
    if device.type == "cpu":
        print("  WARNUNG: kein GPU-Backend. ResNet18 auf CPU braucht pro Epoche")
        print("  Minuten statt Sekunden -- der volle Plan laeuft damit tagelang.")
        print("  Auf AMD unter Windows:  pip install torch-directml")

    payload = json.loads(args.splits.read_text())
    labels, all_folds = payload["labels"], payload["folds"]
    crop_side = load_crop_side(args.prepared)

    rows = []
    for variant in args.variants:
        root = args.prepared / variant
        if not root.exists():
            print(f"  uebersprungen, fehlt: {root}")
            continue
        for k in args.folds:
            for seed in args.seeds:
                print(f"\n  {variant} | fold {k} | seed {seed}")
                res = run_one(root, labels, all_folds[k], k, seed, args, device, crop_side)
                rows.append({"variant": variant, "fold": k, "seed": seed, **res})
                pd.DataFrame(rows).to_csv(args.out, index=False)  # inkrementell sichern

    df = pd.DataFrame(rows)
    if df.empty:
        return

    print("\n" + "=" * 72)
    agg = df.groupby("variant")[["auc", "auc_last", "auc_matched", "auprc",
                                 "sens", "spec", "sens_oracle", "spec_oracle",
                                 "auc_perturbed_mean", "robustness_drop"]]
    print(agg.agg(["mean", "std"]).round(4).to_string())

    resid = df["matched_resid"].dropna()
    if len(resid):
        print(f"\nSelbstkontrolle Matching: Restleak crop_side {resid.min():.3f}-"
              f"{resid.max():.3f} (Soll ~0.500), Kohorte n="
              f"{df['matched_n'].min()}-{df['matched_n'].max()} von "
              f"{df['n_val'].min()}-{df['n_val'].max()} Val-Bildern")
        if (resid - 0.5).abs().max() >= 0.02:
            print("  WARNUNG: Confounder nicht neutralisiert -- auc_matched nicht "
                  "interpretierbar. Caliper verkleinern.")
    sel_gap = (df["auc"] - df["auc_last"]).abs()
    print(f"Selektionseffekt: |bestes Checkpoint - letzte Epoche| im Mittel "
          f"{sel_gap.mean():.4f}, max {sel_gap.max():.4f}")

    for metric in ["auc", "auc_matched"]:
        if df[metric].isna().all():
            continue
        print(f"\nGepaarte Differenzen in '{metric}' (gleicher Fold + Seed):")
        piv = df.pivot_table(index=["fold", "seed"], columns="variant", values=metric)
        for a, b in combinations([v for v in args.variants if v in piv.columns], 2):
            d = (piv[a] - piv[b]).dropna()
            if len(d) == 0:
                continue
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            ci = 1.96 * se
            verdict = "kein Unterschied" if not (abs(d.mean()) > ci) else "Unterschied"
            print(f"  {a} - {b}: {d.mean():+.4f} +/- {ci:.4f} (n={len(d)})  -> {verdict}")

    # Die Streuung ZWISCHEN Seeds unterschaetzt die Unsicherheit: sie enthaelt
    # kein Sampling-Rauschen des Val-Sets. Das Bootstrap-CI der gematchten AUC
    # ist der ehrlichere Vergleichsmassstab fuer die Differenzen darueber.
    if not df["auc_matched"].isna().all():
        w = (df["auc_matched_hi"] - df["auc_matched_lo"]).mean()
        print(f"\nZum Einordnen: mittlere Breite des 95%-CI von auc_matched "
              f"{w:.4f}. Gepaarte Differenzen deutlich darunter sind keine "
              f"belastbare Aussage ueber neue Daten.")

    print(f"\ngespeichert: {args.out}"
          + (f" | Vorhersagen: {args.pred_dir}/" if args.pred_dir else ""))
    print("Das offizielle test/-Holdout erst auswerten, wenn die Variante feststeht.")


if __name__ == "__main__":
    main()
