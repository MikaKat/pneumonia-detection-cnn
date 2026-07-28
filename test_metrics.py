"""Prueft caliper_match, matched_scores und inner_split ohne torch."""
import sys, types, json
import numpy as np, pandas as pd
from pathlib import Path

# --- torch/torchvision stubben, damit der Import ohne GPU-Stack durchlaeuft ---
for name in ["torch", "torch.nn", "torch.optim", "torch.utils", "torch.utils.data",
             "torchvision", "torchvision.transforms", "torchvision.models"]:
    m = types.ModuleType(name); sys.modules[name] = m
sys.modules["torch"].no_grad = lambda: (lambda f: f)
sys.modules["torch"].nn = sys.modules["torch.nn"]
sys.modules["torch.utils.data"].Dataset = object
sys.modules["torch.utils.data"].DataLoader = object
sys.modules["torchvision"].transforms = sys.modules["torchvision.transforms"]
sys.modules["torchvision.models"].resnet18 = None
sys.modules["torchvision.models"].ResNet18_Weights = None

import train_compare as tc
from sklearn.metrics import roc_auc_score

fail = []

# 1) caliper_match gegen eine Brute-Force-Referenz
def ref_match(y, x, cal):
    a = list(np.where(y == 0)[0]); b = list(np.where(y == 1)[0])
    if len(a) > len(b): a, b = b, a
    a = sorted(a, key=lambda i: x[i]); free = set(b); ka, kb = [], []
    for i in a:
        cand = sorted(free, key=lambda j: (abs(x[j] - x[i]), j))
        if cand and abs(x[cand[0]] - x[i]) <= cal:
            free.discard(cand[0]); ka.append(i); kb.append(cand[0])
    return ka, kb

rng = np.random.default_rng(0)
for trial in range(200):
    n = rng.integers(10, 120)
    y = rng.integers(0, 2, n)
    x = np.round(rng.normal(size=n), 2)      # bewusst viele Bindungen
    cal = float(rng.choice([0.0, 0.05, 0.2, 1.0]))
    got = tc.caliper_match(y, x, cal)
    ka, kb = ref_match(y, x, cal)
    if len(got) < len(ka) + len(kb):          # greedy darf nicht schlechter sein
        fail.append(f"weniger Paare als Referenz trial{trial}: {len(got)} < {len(ka)+len(kb)}")
        break
    if len(got):
        yy = y[got]; h = len(got) // 2
        if (yy == 0).sum() != (yy == 1).sum():
            fail.append(f"unbalanciert trial{trial}"); break
        if len(set(got.tolist())) != len(got):
            fail.append(f"Index doppelt verwendet trial{trial}"); break
        if (y[got[:h]] == y[got[h:]]).any():
            fail.append(f"Paar mit gleichem Label trial{trial}"); break
        dmax = np.abs(x[got[:h]] - x[got[h:]]).max()
        if dmax > cal + 1e-12:
            fail.append(f"Paar ausserhalb Caliper trial{trial}: {dmax} > {cal}"); break
print("1) caliper_match, 200 Zufallsfaelle (Balance, Eindeutigkeit, Caliper, >=Referenz):",
      "OK" if not fail else fail[-1])

# 2) matched_scores auf den echten Folds: Restleak muss ~0.50 sein
pl = json.loads(Path("splits.json").read_text())
lab = pl["labels"]
d = pd.read_csv("data/prepared/crop_log.csv")
d["key"] = [str(Path(f).with_suffix("")).replace("\\", "/") for f in d["file"]]
cs_map = dict(zip(d["key"], d["crop_side"]))
print("2) matched_scores auf echten Folds (Caliper 0.01):")
for k in range(5):
    f = pl["folds"][k]
    y = np.array([lab[x] for x in f["val"]])
    cs = np.array([cs_map[str(Path(x).with_suffix("")).replace("\\", "/")] for x in f["val"]])
    p = 1 / (1 + np.exp(-(1200 - cs) / 200))          # reiner Confounder-"Klassifikator"
    r = tc.matched_scores(y, p, cs, 0.01, n_boot=400, seed=k)
    ok = abs(r["matched_resid"] - 0.5) < 0.02
    print(f"   fold{k}: n={r['matched_n']:4d}/{len(y)} resid={r['matched_resid']:.3f} "
          f"| Shortcut-Modell roh {roc_auc_score(y, p):.3f} -> gematcht "
          f"{r['auc_matched']:.3f} [{r['auc_matched_lo']:.3f}-{r['auc_matched_hi']:.3f}]"
          f"{'' if ok else '   RESID FAIL'}")
    if not ok: fail.append(f"resid fold{k}")

# 3) inner_split: Gruppen disjunkt, kein Val-Kontakt, Stratifizierung haelt
print("3) inner_split:")
for k in [0, 2]:
    for seed in [0, 1]:
        f = pl["folds"][k]
        fit, sel = tc.inner_split(f["train"], lab, seed, 6)
        gf = {tc.parse_record(Path(x.replace(chr(92), "/")))["group"] for x in fit}
        gs = {tc.parse_record(Path(x.replace(chr(92), "/")))["group"] for x in sel}
        gv = {tc.parse_record(Path(x.replace(chr(92), "/")))["group"] for x in f["val"]}
        bad = (gf & gs) or ((gf | gs) & gv) or (set(fit) & set(sel))
        pr_f = np.mean([lab[x] for x in fit]); pr_s = np.mean([lab[x] for x in sel])
        print(f"   fold{k} seed{seed}: fit={len(fit)} sel={len(sel)} "
              f"({len(sel)/(len(fit)+len(sel)):.0%}) pos fit={pr_f:.3f} sel={pr_s:.3f} "
              f"| Leak: {'NEIN' if not bad else 'JA <-- FAIL'}")
        if bad: fail.append(f"inner_split leak fold{k} seed{seed}")
        if len(fit) + len(sel) != len(f["train"]): fail.append("inner_split verliert Bilder")

print("\n" + ("ALLE CHECKS OK" if not fail else "FEHLER: " + "; ".join(fail)))
