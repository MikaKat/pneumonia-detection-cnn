"""
Sanity-Lauf auf RSNA: EIN Fold, EIN Modell, alle Kontrollen.

Zweck ist ausdruecklich nicht der beste Wert, sondern eine belastbare erste
Zahl und die Antwort auf vier Fragen:

  1. Schlaegt das Modell die Nur-Header-Baseline von **0,729**? Alles darunter
     heisst: es hat weniger gelernt als ein Klassifikator, der kein Bild sieht.
  2. Schlaegt es sie **innerhalb einer Projektion**? Dort liegt die Baseline bei
     0,553 (AP) / 0,559 (PA). Das ist die eigentliche Frage -- die Gesamt-AUC
     enthaelt den AP/PA-Effekt, die geschichtete nicht.
  3. **Zeigt Grad-CAM auf die Pathologie?** RSNA hat Bounding Boxes, damit ist
     das erstmals messbar statt Ansichtssache. Das war die Ausgangsfrage des
     ganzen Projekts.
  4. **Liest das Modell die eingebrannten Marker?** Auf AP-Bildern steht
     sichtbar "PORTABLE". Die Ecken-Ablation beantwortet das.

Uebernommen aus Phase 3, weil dort teuer gelernt:
  * Checkpoint UND Schwelle kommen von einem inneren, patientengruppierten
    Selektions-Split. Das aeussere Val wird nur berichtet, nie optimiert.
  * `auc_last` und `*_oracle` laufen als optimistische Referenz mit, damit die
    Luecke sichtbar bleibt.
  * Alle Vorhersagen landen als CSV auf der Platte. Auf Kermany kostete jede
    Nachfrage sonst einen kompletten Retraining-Lauf.

Neu gegenueber Phase 3:
  * Kein Caliper-Matching. Der Confounder ist hier binaer und exakt bekannt,
    also wird geschichtet -- das kostet kein einziges Bild, waehrend das
    Matching auf Kermany zwei Drittel der Daten verwarf.
  * Kein `RandomResolution`. Der Jitter war gegen den Kermany-Zoom-Confounder
    gebaut; hier sind alle Bilder gleich gross, er wuerde nur Rauschen addieren.

CLI:
  python rsna_train.py --fold 0 --epochs 8 --batch 16 --workers 0
"""

from __future__ import annotations

import argparse
import json
import random
import time
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

IMNET_MEAN, IMNET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BOX_SPACE = 1024          # Bounding Boxes liegen im Original-DICOM-Raster


# --------------------------------------------------------------------------
# Daten
# --------------------------------------------------------------------------

class RsnaDataset(Dataset):
    """IDs statt Pfade -- den Pfad baut erst der Loader. Siehe rsna_splits.py."""

    def __init__(self, root: Path, ids: list[str], labels: dict[str, int], tf):
        self.root, self.ids, self.labels, self.tf = root, ids, labels, tf

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]
        img = Image.open(self.root / f"{pid}.png").convert("L")
        return self.tf(img), float(self.labels[pid])


class MaskCorners:
    """Setzt die vier Bildecken auf den Median -- Ablation gegen Marker-Lesen.

    Auf den AP-Aufnahmen stehen "PORTABLE", Seitenmarker und Pfeile im Bild.
    Das ist ein direkter visueller Proxy fuer die Aufnahmeart, und die
    Aufnahmeart ist der gesamte Confounder. Ein grober Statistiktest (Anteil
    heller Pixel in den Ecken) findet nichts, aber ein Faltungsnetz liest
    Schrift besser als eine Helligkeitsschwelle -- also wird es ausprobiert
    statt bewertet.

    Bewusst der Median und nicht Schwarz: eine schwarze Flaeche ist selbst ein
    auffaelliges Merkmal und wuerde eine neue Kante einfuehren.
    """

    def __init__(self, frac: float = 0.18):
        self.frac = frac

    def __call__(self, img: Image.Image) -> Image.Image:
        a = np.asarray(img).copy()
        k = int(min(a.shape[:2]) * self.frac)
        med = int(np.median(a))
        a[:k, :k] = med; a[:k, -k:] = med; a[-k:, :k] = med; a[-k:, -k:] = med
        return Image.fromarray(a)


def build_transforms(size: int, train: bool):
    # KEIN horizontales Spiegeln: erzeugt Situs inversus, vertauscht die
    # Herzsilhouette und widerspricht dem Seitenmarker im Bild.
    base = [T.Grayscale(num_output_channels=3), T.ToTensor(),
            T.Normalize(IMNET_MEAN, IMNET_STD)]
    if not train:
        return T.Compose([T.Resize((size, size))] + base)
    return T.Compose([
        T.Resize((size, size)),
        T.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.93, 1.07)),
        T.ColorJitter(brightness=0.15, contrast=0.15),
    ] + base)


PERTURBATIONS = {
    "clean":       lambda s: [T.Resize((s, s))],
    "corners":     lambda s: [MaskCorners(), T.Resize((s, s))],
    "zoom_in":     lambda s: [T.Resize((int(s * 1.15),) * 2), T.CenterCrop(s)],
    "shift":       lambda s: [T.Resize((s, s)), T.RandomAffine(0, translate=(0.08, 0.08))],
    "rotate":      lambda s: [T.Resize((s, s)), T.RandomRotation(12)],
    "low_contr":   lambda s: [T.Resize((s, s)), T.ColorJitter(contrast=(0.6, 0.6))],
    "bright":      lambda s: [T.Resize((s, s)), T.ColorJitter(brightness=(1.35, 1.35))],
    "blur":        lambda s: [T.Resize((s, s)), T.GaussianBlur(5, sigma=1.6)],
    "lowres":      lambda s: [T.Resize((int(s * 0.45),) * 2), T.Resize((s, s))],
}


def perturbed_transform(size: int, name: str):
    return T.Compose(PERTURBATIONS[name](size) +
                     [T.Grayscale(num_output_channels=3), T.ToTensor(),
                      T.Normalize(IMNET_MEAN, IMNET_STD)])


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def pick_device(name: str):
    """DirectML ist auf dieser Hardware der einzige GPU-Weg (RX 5500 XT, RDNA1)."""
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


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tpr = np.cumsum(ys) / max(ys.sum(), 1)
    fpr = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
    return float(ps[int(np.argmax(tpr - fpr))])


def scores(y: np.ndarray, p: np.ndarray, thr: float | None = None) -> dict:
    """Rangmetriken plus Sens/Spez an einer VORGEGEBENEN Schwelle.

    `*_oracle` sucht die Schwelle auf demselben Satz, auf dem berichtet wird --
    absichtlich mitgeschrieben als optimistisches Gegenstueck, damit die Luecke
    zur ehrlichen Zahl sichtbar bleibt.
    """
    out = {"auc": float(roc_auc_score(y, p)),
           "auprc": float(average_precision_score(y, p))}
    t_or = youden_threshold(y, p)
    out["sens_oracle"] = float(((p >= t_or) & (y == 1)).sum() / max((y == 1).sum(), 1))
    out["spec_oracle"] = float(((p < t_or) & (y == 0)).sum() / max((y == 0).sum(), 1))
    t = t_or if thr is None else thr
    out["thr"] = float(t)
    out["sens"] = float(((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1))
    out["spec"] = float(((p < t) & (y == 0)).sum() / max((y == 0).sum(), 1))
    return out


def stratified_scores(y: np.ndarray, p: np.ndarray, vp: np.ndarray,
                      thr: float, thr_by_view: dict[str, float] | None = None
                      ) -> dict:
    """AUC je Projektion, plus Sens/Spez an globaler UND schichteigener Schwelle.

    Die Gesamt-AUC enthaelt den AP/PA-Effekt (Header-Baseline 0,729). Innerhalb
    einer Projektion faellt der weg, dort liegt die Baseline bei ~0,556. Nur
    diese geschichtete Zahl sagt etwas ueber Radiologie aus.

    Warum zusaetzlich zwei Schwellen: der erste Lauf hatte bei praktisch
    identischer AUC (0,818 AP gegen 0,824 PA) an EINER Schwelle Sens 0,839 in
    AP und 0,498 in PA. Dieselbe Zahl verhaelt sich in den beiden Projektionen
    wie zwei verschiedene Tests -- in PA-Aufnahmen waere die Haelfte der
    Pneumonien uebersehen worden. Das ist kein Modellfehler, sondern der
    Praevalenzunterschied (0,383 gegen 0,093), der ueber eine feste Schwelle in
    Sensitivitaet umschlaegt. Beides wird berichtet, damit der Effekt sichtbar
    bleibt statt weggerechnet zu werden.
    """
    out = {}
    for v in ("AP", "PA"):
        m = vp == v
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        pos, neg = y[m] == 1, y[m] == 0
        out[f"auc_{v}"] = float(roc_auc_score(y[m], p[m]))
        out[f"n_{v}"] = int(m.sum())
        out[f"pos_{v}"] = float(y[m].mean())
        out[f"sens_{v}"] = float(((p[m] >= thr) & pos).sum() / max(pos.sum(), 1))
        out[f"spec_{v}"] = float(((p[m] < thr) & neg).sum() / max(neg.sum(), 1))
        if thr_by_view and v in thr_by_view:
            t = thr_by_view[v]
            out[f"thr_{v}"] = float(t)
            out[f"sens_{v}_strat"] = float(((p[m] >= t) & pos).sum() / max(pos.sum(), 1))
            out[f"spec_{v}_strat"] = float(((p[m] < t) & neg).sum() / max(neg.sum(), 1))

    if "auc_AP" in out and "auc_PA" in out:
        # gewichtetes Mittel der Schichten: die AUC, die uebrig bliebe, wenn
        # AP und PA gleich haeufig waeren -- der confounderbereinigte Wert
        w = np.array([out["n_AP"], out["n_PA"]], float)
        out["auc_stratified"] = float(
            (out["auc_AP"] * w[0] + out["auc_PA"] * w[1]) / w.sum())
        # Das direkte Mass fuer das Problem: wie weit klaffen die
        # Sensitivitaeten zwischen den Projektionen auseinander?
        out["sens_gap"] = float(abs(out["sens_AP"] - out["sens_PA"]))
        if "sens_AP_strat" in out and "sens_PA_strat" in out:
            out["sens_gap_strat"] = float(
                abs(out["sens_AP_strat"] - out["sens_PA_strat"]))
    return out


def inner_split(ids: list[str], labels, vp: dict[str, str], seed: int,
                n_splits: int) -> tuple[list[str], list[str]]:
    """Teilt fold["train"] in Fit- und Selektionsteil, geschichtet wie aussen.

    Warum ueberhaupt: vorher wurde das Checkpoint per AUC auf dem AEUSSEREN Val
    gewaehlt und dieselbe AUC berichtet -- jede Zahl war damit ein Maximum ueber
    alle Epochen auf den Berichtsdaten. Auf Kermany versteckte die Decke das;
    bei AUC ~0,85 verschiebt es die Zahl um die Groessenordnung der Effekte,
    die man messen will.

    Geschichtet wird auch hier nach label x ViewPosition, sonst weicht die
    AP/PA-Quote des Selektions-Splits vom Val ab und die Schwelle passt nicht.
    """
    strat = np.array([f"{labels[i]}|{vp[i]}" for i in ids])
    g = np.array(ids)                      # ein Bild je Patient
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fit_i, sel_i = next(iter(sgkf.split(np.zeros(len(ids)), strat, g)))
    assert not (set(g[fit_i]) & set(g[sel_i])), "Gruppen-Leak im inneren Split!"
    return [ids[i] for i in fit_i], [ids[i] for i in sel_i]


# --------------------------------------------------------------------------
# Grad-CAM gegen Bounding Boxes
# --------------------------------------------------------------------------

def load_boxes(csv_dir: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    df = pd.read_csv(Path(csv_dir) / "stage_2_train_labels.csv")
    df = df[df["Target"] == 1].dropna(subset=["x", "y", "width", "height"])
    out: dict[str, list] = {}
    for pid, x, y, w, h in df[["patientId", "x", "y", "width", "height"]].values:
        out.setdefault(pid, []).append((float(x), float(y), float(w), float(h)))
    return out


def cam_vs_boxes(model, root: Path, ids: list[str], boxes: dict, size: int,
                 n: int, seed: int) -> tuple[dict, pd.DataFrame]:
    """Misst, ob die Heatmap auf das Infiltrat zeigt.

    Zwei Masse, beide gegen eine Zufallsbaseline:

      hit    Liegt das Maximum der Heatmap in einer Box? ("pointing game")
             Zufallsbaseline = Flaechenanteil der Boxen.
      mass   Welcher Anteil der Heatmap-Masse liegt in den Boxen?
             Zufallsbaseline ebenfalls der Flaechenanteil.

    Der Flaechenanteil MUSS mitberichtet werden. Die Boxen decken einen
    erheblichen Teil des Bildes ab; eine Trefferquote von 0,6 klingt gut und
    waere bei 0,55 Flaechenanteil nahezu nichts. Ohne diese Baseline ist die
    Zahl wertlos -- genau der Fehler, den Grad-CAM-Abbildungen in Praesentationen
    ueblicherweise machen.

    Laeuft auf der CPU: Grad-CAM braucht Rueckwaertspfad durch Hooks, und das
    ist unter DirectML weder schnell noch zuverlaessig.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

    rng = np.random.default_rng(seed)
    pos = [i for i in ids if i in boxes]
    if not pos:
        return {}, pd.DataFrame()
    pick = rng.choice(pos, min(n, len(pos)), replace=False)

    m = model.to("cpu").eval()
    cam = GradCAM(model=m, target_layers=[m.layer4[-1]])
    tf = build_transforms(size, False)
    s = size / BOX_SPACE

    rows = []
    for j, pid in enumerate(pick, 1):
        img = Image.open(root / f"{pid}.png").convert("L")
        x = tf(img).unsqueeze(0)
        heat = cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(1)])[0]
        heat = np.clip(heat, 0, None)
        if heat.sum() <= 0:
            continue

        mask = np.zeros_like(heat, bool)
        for bx, by, bw, bh in boxes[pid]:
            x0, y0 = int(bx * s), int(by * s)
            x1, y1 = int((bx + bw) * s), int((by + bh) * s)
            mask[max(y0, 0):y1, max(x0, 0):x1] = True

        area = float(mask.mean())
        yx = np.unravel_index(int(np.argmax(heat)), heat.shape)
        rows.append({"patientId": pid, "hit": bool(mask[yx]),
                     "mass": float(heat[mask].sum() / heat.sum()),
                     "area": area, "n_boxes": len(boxes[pid])})
        if j % 100 == 0:
            print(f"      Grad-CAM {j}/{len(pick)}")

    d = pd.DataFrame(rows)
    if d.empty:
        return {}, d
    res = {
        "cam_n": int(len(d)),
        "cam_hit": float(d["hit"].mean()),
        "cam_mass": float(d["mass"].mean()),
        "cam_area_baseline": float(d["area"].mean()),
        "cam_hit_lift": float(d["hit"].mean() - d["area"].mean()),
        "cam_mass_lift": float(d["mass"].mean() - d["area"].mean()),
    }
    return res, d


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, default=Path("data/rsna/png512"))
    p.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=0,
                   help="unter Windows 0 lassen: spawn importiert Torch pro Worker neu")
    p.add_argument("--inner-splits", type=int, default=6)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "directml", "cpu"])
    p.add_argument("--cam-n", type=int, default=300, help="0 = Grad-CAM ueberspringen")
    p.add_argument("--out", type=Path, default=Path("results_rsna.csv"))
    p.add_argument("--pred-dir", type=Path, default=Path("predictions_rsna"))
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    sp = json.loads(args.splits.read_text())
    labels = {k: int(v) for k, v in sp["labels"].items()}
    vpmap = sp["viewpos"]
    fold = sp["folds"][args.fold]
    device, pin = pick_device(args.device)

    fit_ids, sel_ids = inner_split(fold["train"], labels, vpmap,
                                   args.seed, args.inner_splits)
    val_ids = fold["val"]
    y_fit = np.array([labels[i] for i in fit_ids])

    print(f"\nFold {args.fold}, Seed {args.seed}, Device {device}")
    print(f"  fit {len(fit_ids)} (pos {y_fit.mean():.3f}) | sel {len(sel_ids)} "
          f"| val {len(val_ids)}")
    print(f"  Zielmarken: Gesamt-AUC > 0.729 (Header-Baseline), "
          f"je Projektion > ~0.556")
    if device.type == "cpu":
        print("  WARNUNG: CPU. Auf AMD unter Windows:  pip install torch-directml")

    tr = DataLoader(RsnaDataset(args.images, fit_ids, labels,
                               build_transforms(args.size, True)),
                    batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=pin, drop_last=True)
    sel = DataLoader(RsnaDataset(args.images, sel_ids, labels,
                                build_transforms(args.size, False)),
                     batch_size=args.batch * 2, num_workers=args.workers)
    va = DataLoader(RsnaDataset(args.images, val_ids, labels,
                               build_transforms(args.size, False)),
                    batch_size=args.batch * 2, num_workers=args.workers)

    model = make_model(device)
    # Positivrate 0.225 -- das Ungleichgewicht kippt gegenueber Kermany (0.74)
    # in die andere Richtung, pos_weight also > 1 statt < 1.
    pos_weight = torch.tensor([(y_fit == 0).sum() / max((y_fit == 1).sum(), 1)],
                              dtype=torch.float32, device=device)
    print(f"  pos_weight {pos_weight.item():.2f}")
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
            opt.step(); sched.step()
        ps, ys = predict(model, sel, device)
        a = roc_auc_score(ys, ps)
        if a > best_sel:
            best_sel, best_ep = a, ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        dt = time.time() - t0
        print(f"  epoch {ep + 1}/{args.epochs}  sel AUC {a:.4f}  "
              f"[{dt:.0f}s, Rest ~{dt * (args.epochs - ep - 1) / 60:.0f} min]")

    p_last, y = predict(model, va, device)
    auc_last = float(roc_auc_score(y, p_last))

    model.load_state_dict(best_state)
    p_sel, y_sel = predict(model, sel, device)
    thr = youden_threshold(y_sel, p_sel)          # Schwelle NICHT vom Berichtsatz
    # ... und je Projektion ebenfalls auf dem Selektions-Split, nicht auf dem Val.
    # Eine auf dem Berichtsatz gesuchte Schichtschwelle waere derselbe
    # Zirkelschluss wie die globale und wuerde den Gewinn ueberzeichnen.
    vp_sel = np.array([vpmap[i] for i in sel_ids])
    thr_by_view = {v: youden_threshold(y_sel[vp_sel == v], p_sel[vp_sel == v])
                   for v in ("AP", "PA")
                   if (vp_sel == v).sum() >= 50
                   and len(np.unique(y_sel[vp_sel == v])) > 1}
    p_val, y = predict(model, va, device)

    vp = np.array([vpmap[i] for i in val_ids])
    res = scores(y, p_val, thr)
    res.update(stratified_scores(y, p_val, vp, thr, thr_by_view))
    res.update({"fold": args.fold, "seed": args.seed, "epochs": args.epochs,
                "auc_last": auc_last, "auc_sel": float(best_sel),
                "best_epoch": best_ep + 1, "n_fit": len(fit_ids),
                "n_sel": len(sel_ids), "n_val": len(val_ids)})

    print(f"\n  AUC gesamt      {res['auc']:.4f}   (letzte Epoche {auc_last:.4f}, "
          f"Header-Baseline 0.729)")
    for v in ("AP", "PA"):
        if f"auc_{v}" in res:
            print(f"  AUC nur {v}      {res[f'auc_{v}']:.4f}   "
                  f"(n={res[f'n_{v}']}, pos={res[f'pos_{v}']:.3f}, "
                  f"Baseline ~0.556)")
    if "auc_stratified" in res:
        print(f"  AUC geschichtet {res['auc_stratified']:.4f}  <-- die ehrliche Zahl")
    print(f"  Sens {res['sens']:.3f} / Spez {res['spec']:.3f} "
          f"(Oracle {res['sens_oracle']:.3f}/{res['spec_oracle']:.3f})")

    # Der Kern: verhaelt sich EINE Schwelle in beiden Projektionen gleich?
    if "sens_gap" in res:
        print(f"\n  Schwelle            {'global':>22}   {'je Projektion':>22}")
        for v in ("AP", "PA"):
            g = f"Sens {res[f'sens_{v}']:.3f} Spez {res[f'spec_{v}']:.3f}"
            s = (f"Sens {res[f'sens_{v}_strat']:.3f} Spez {res[f'spec_{v}_strat']:.3f}"
                 f" @{res[f'thr_{v}']:.3f}" if f"sens_{v}_strat" in res else "-")
            print(f"    {v:<16}{g:>22}   {s:>22}")
        line = f"    {'Sens-Luecke':<16}{res['sens_gap']:>22.3f}"
        if "sens_gap_strat" in res:
            line += f"   {res['sens_gap_strat']:>22.3f}"
        print(line)
        print("    Eine feste Schwelle ist bei ungleicher Praevalenz (0.383 vs 0.093)")
        print("    in den beiden Projektionen faktisch ein anderer Test.")

    # ---- Stoerungen, allen voran die Ecken-Ablation --------------------
    preds = {"patientId": list(val_ids), "y": y.tolist(), "viewpos": vp.tolist(),
             "p_clean": p_val.tolist(), "p_last_epoch": p_last.tolist()}
    print()
    for name in [n for n in PERTURBATIONS if n != "clean"]:
        torch.manual_seed(args.seed); random.seed(args.seed)
        ds = RsnaDataset(args.images, val_ids, labels,
                         perturbed_transform(args.size, name))
        pp, yy = predict(model, DataLoader(ds, batch_size=args.batch * 2,
                                           num_workers=args.workers), device)
        res[f"auc_{name}"] = float(roc_auc_score(yy, pp))
        preds[f"p_{name}"] = pp.tolist()
        tag = "  <-- Marker-Ablation" if name == "corners" else ""
        print(f"  Stoerung {name:<10} AUC {res[f'auc_{name}']:.4f}  "
              f"({res[f'auc_{name}'] - res['auc']:+.4f}){tag}")

    # ---- Grad-CAM gegen Bounding Boxes ---------------------------------
    cam_df = pd.DataFrame()
    if args.cam_n:
        print(f"\n  Grad-CAM auf {args.cam_n} positiven Val-Bildern (CPU)...")
        boxes = load_boxes(args.csv)
        cam_res, cam_df = cam_vs_boxes(model, args.images, val_ids, boxes,
                                       args.size, args.cam_n, args.seed)
        res.update(cam_res)
        if cam_res:
            print(f"  Treffer  {cam_res['cam_hit']:.3f}  vs Zufall "
                  f"{cam_res['cam_area_baseline']:.3f}  "
                  f"(Vorsprung {cam_res['cam_hit_lift']:+.3f})")
            print(f"  Masse    {cam_res['cam_mass']:.3f}  vs Zufall "
                  f"{cam_res['cam_area_baseline']:.3f}  "
                  f"(Vorsprung {cam_res['cam_mass_lift']:+.3f})")
            print("  Ohne die Zufallsbaseline ist die Trefferquote bedeutungslos:")
            print("  die Boxen decken einen erheblichen Teil des Bildes ab.")

    # ---- Speichern ------------------------------------------------------
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds).to_csv(
        args.pred_dir / f"rsna_f{args.fold}_s{args.seed}.csv", index=False)
    # Auch die Selektions-Vorhersagen: ohne sie laesst sich jede Frage zur
    # Schwelle spaeter nur als Oracle beantworten (Schwelle auf dem Berichtsatz
    # gesucht = zu optimistisch). Genau daran fehlte es nach dem ersten Lauf.
    pd.DataFrame({"patientId": sel_ids, "y": y_sel.tolist(),
                  "viewpos": vp_sel.tolist(), "p_sel": p_sel.tolist()}).to_csv(
        args.pred_dir / f"sel_f{args.fold}_s{args.seed}.csv", index=False)
    if not cam_df.empty:
        cam_df.to_csv(args.pred_dir / f"cam_f{args.fold}_s{args.seed}.csv", index=False)
    torch.save(best_state, f"checkpoints/rsna_f{args.fold}_s{args.seed}.pth")

    # NICHT anhaengen, sondern einlesen-zusammenfuehren-neuschreiben.
    # Ein blosses mode="a" schreibt die Werte in der Reihenfolge des AKTUELLEN
    # Laufs unter eine Kopfzeile, die aus einem frueheren stammt. Sobald sich
    # das Metrikset aendert -- hier kamen thr_AP/sens_AP_strat/... hinzu --
    # stehen 49 Werte unter 41 Spaltennamen. Die Datei ist dann nicht kaputt im
    # Sinne von unlesbar, sondern schlimmer: still verschoben.
    row = pd.DataFrame([res])
    if args.out.exists():
        old = pd.read_csv(args.out)
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(args.out, index=False)
    print(f"\ngespeichert: {args.out}, {args.pred_dir}/, "
          f"checkpoints/rsna_f{args.fold}_s{args.seed}.pth")


if __name__ == "__main__":
    main()
