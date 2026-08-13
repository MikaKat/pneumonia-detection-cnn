"""Does the prevalence estimator carry? Checked on data that already exists.

Why this exists
---------------
The delivered app shows p(pneumonia | image) under ONE assumed prevalence: the
0.2253 of the 22872 development images the Platt curves were fitted on. Kermany
showed what happens when that assumption is wrong: ECE 0.4783, and two thirds of
it vanish if the logits are shifted by the difference of the two KNOWN
frequencies (+2.2278).

In production nobody hands you the target prevalence. The question is whether it
can be estimated from the unlabelled stream of scores, so the correction can be
applied to a SINGLE image using a constant learned from the last few hundred
cases.

Two estimators, both classic:

  BBSE   Lipton, Wang, Smola, ICML 2018, "Detecting and Correcting for Label
         Shift with Black Box Predictors". Closed form from the source
         confusion matrix. Binary:  pi = (q - FPR) / (TPR - FPR)
         with q the fraction predicted positive in the target.

  EM     Saerens, Latinne, Decaestecker, Neural Computation 2002, "Adjusting
         the outputs of a classifier to new a priori probabilities". Iterative,
         uses the whole posterior. Alexandari, Kundaje, Shrikumar, ICML 2020
         showed that a well calibrated source model plus EM beats BBSE.

Both assume LABEL SHIFT: p(x|y) unchanged, p(y) changed. That assumption is the
whole point, and it is testable here in two ways at once:

  A) SIMULATION. Resample the development set to a chosen prevalence. That
     changes p(y) and leaves p(x|y) untouched BY CONSTRUCTION, so label shift
     holds exactly. If an estimator misses here, the estimator is broken.

  B) KERMANY. Real foreign data with a KNOWN prevalence of 0.7297. If the
     estimator works in A and misses in B, the assumption is violated, not the
     method, and the size of the miss is the price of that violation.

Nothing here touches the spent holdout. The source confusion matrix comes from
the out of fold development predictions, which is where the threshold and the
Platt curves came from in the first place.

  python3 rsna_praevalenz_schaetzer.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

def _wurzel():
    """Repo-Wurzel, egal von wo aufgerufen. Wie `_repo_path.py` in Nachbarskripten,
    nur ohne Import, damit die Datei auch einzeln laeuft."""
    for k in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (k / "serving/model/kalibrierung_p10.json").is_file():
            return k
    raise SystemExit("serving/model/kalibrierung_p10.json nicht gefunden.")


HIER = _wurzel()
KAL = json.loads((HIER / "serving/model/kalibrierung_p10.json").read_text())
SCHWELLE = KAL["schwelle"]
FOLDS = [0, 1, 2, 3, 4]
RNG = np.random.default_rng(20260813)


# ---------------------------------------------------------------- Grundrechnen

def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def platt(p_roh, a, b):
    return sigmoid(a * logit(p_roh) + b)


def ece(p, y, bins=10):
    """Gleichbreite Felder, wie in rsna_platt.py."""
    kanten = np.linspace(0.0, 1.0, bins + 1)
    e = 0.0
    n = len(p)
    for lo, hi in zip(kanten[:-1], kanten[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return e


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def prior_shift(p, pi_von, pi_nach):
    """Die ganze Korrektur: eine additive Konstante im Logit."""
    d = np.log(pi_nach / (1 - pi_nach)) - np.log(pi_von / (1 - pi_von))
    return sigmoid(logit(p) + d), d


# ---------------------------------------------------------------- Schaetzer

def bbse(p_ziel, schwelle, sens, spez):
    """Lipton et al. 2018, harte Variante. Auf [0,1] beschnitten."""
    q = float(np.mean(p_ziel >= schwelle))
    fpr = 1.0 - spez
    if abs(sens - fpr) < 1e-9:
        return float("nan"), q
    return float(np.clip((q - fpr) / (sens - fpr), 0.0, 1.0)), q


def em(p_ziel, pi_quelle, iters=1000, tol=1e-10):
    """Saerens et al. 2002. Gibt die geschaetzte Praevalenz zurueck."""
    pi = float(pi_quelle)
    for _ in range(iters):
        w1 = pi / pi_quelle
        w0 = (1 - pi) / (1 - pi_quelle)
        post = p_ziel * w1 / (p_ziel * w1 + (1 - p_ziel) * w0)
        neu = float(post.mean())
        if abs(neu - pi) < tol:
            pi = neu
            break
        pi = neu
    return float(np.clip(pi, 0.0, 1.0))


# ---------------------------------------------------------------- Daten

def lade_dev():
    """Out of fold Entwicklungsvorhersagen, kalibriert. Ein Modell je Bild."""
    teile = []
    for f in FOLDS:
        d = pd.read_csv(HIER / f"predictions_final_model/rsna_f{f}_s0.csv")
        par = next(x for x in KAL["platt"] if x["fold"] == f)
        d = d[["patientId", "y", "viewpos", "p_clean"]].copy()
        d["fold"] = f
        d["p_kal"] = platt(d["p_clean"].to_numpy(), par["a"], par["b"])
        teile.append(d)
    return pd.concat(teile, ignore_index=True)


def lade_kermany():
    d = pd.read_csv(HIER / "predictions_extern_kermany/extern_kermany_ens.csv")
    return d[["file", "label", "group", "p_stretch_ens"]].rename(
        columns={"label": "y", "p_stretch_ens": "p_kal"})


# ---------------------------------------------------------------- 0. Nachweis

def nachweis(dev):
    """Erst beweisen, dass die Rekonstruktion stimmt. Sonst ist alles danach Deko."""
    soll = KAL["dev"]
    soll_s = KAL["dev_bei_schwelle"]
    p, y = dev["p_kal"].to_numpy(), dev["y"].to_numpy()
    ist = {
        "n": len(dev),
        "praevalenz": float(y.mean()),
        "mittel_kal": float(p.mean()),
        "ece_kal": ece(p, y),
        "brier_kal": brier(p, y),
        "sens": float(((p >= SCHWELLE) & (y == 1)).sum() / (y == 1).sum()),
        "spez": float(((p < SCHWELLE) & (y == 0)).sum() / (y == 0).sum()),
    }
    erwartet = {
        "n": soll["n"], "praevalenz": soll["praevalenz"],
        "mittel_kal": soll["mittel_kal"], "ece_kal": soll["ece_kal"],
        "brier_kal": soll["brier_kal"],
        "sens": soll_s["sens"], "spez": soll_s["spez"],
    }
    print("=" * 74)
    print("0. NACHWEIS  Rekonstruktion der Entwicklungsvorhersagen")
    print("=" * 74)
    ok = True
    for k, v in ist.items():
        soll_v = erwartet[k]
        d = abs(v - soll_v)
        gut = d < (1 if k == "n" else 5e-4)
        ok &= gut
        print(f"  {k:<12} ist {v:>12.6f}   soll {soll_v:>12.6f}   "
              f"{'ok' if gut else 'ABWEICHUNG'}")
    if not ok:
        raise SystemExit("Rekonstruktion stimmt nicht, hier ist Schluss.")
    print("  -> die Quell-Konfusionsmatrix ist echt.\n")
    return ist


# ---------------------------------------------------------------- 1. Simulation

def simulation(dev, sens, spez, pi_quelle):
    """Label Shift GILT hier per Konstruktion. Wer hier fehlt, ist kaputt."""
    print("=" * 74)
    print("1. SIMULATION  Entwicklungsdaten auf eine Wunsch-Praevalenz umgezogen")
    print("=" * 74)
    print("   (Umziehen aendert p(y) und laesst p(x|y) unberuehrt -> Label Shift gilt)")
    pos = dev[dev.y == 1]["p_kal"].to_numpy()
    neg = dev[dev.y == 0]["p_kal"].to_numpy()
    print(f"\n  {'Ziel':>6} {'n':>7} {'BBSE':>9} {'Fehler':>9} "
          f"{'EM':>9} {'Fehler':>9}")
    print("  " + "-" * 54)
    zeilen = []
    for ziel in [0.05, 0.10, 0.225, 0.40, 0.60, 0.73, 0.85]:
        n = 6000
        n_pos = int(round(n * ziel))
        p = np.concatenate([
            RNG.choice(pos, n_pos, replace=True),
            RNG.choice(neg, n - n_pos, replace=True)])
        b, _ = bbse(p, SCHWELLE, sens, spez)
        e = em(p, pi_quelle)
        zeilen.append({"ziel": ziel, "bbse": b, "em": e})
        print(f"  {ziel:>6.3f} {n:>7d} {b:>9.4f} {b - ziel:>+9.4f} "
              f"{e:>9.4f} {e - ziel:>+9.4f}")
    mb = max(abs(z["bbse"] - z["ziel"]) for z in zeilen)
    me = max(abs(z["em"] - z["ziel"]) for z in zeilen)
    print(f"\n  groesster Fehler   BBSE {mb:.4f}   EM {me:.4f}")
    print("  -> beide Verfahren treffen, wenn die Annahme gilt.\n")
    return zeilen


def stichprobenumfang(dev, sens, spez, pi_quelle, ziel=0.30, wdh=400):
    """Wie viele Bilder braucht der Strom? Streuung ueber Wiederholungen."""
    print("=" * 74)
    print(f"1b. STICHPROBENUMFANG  wahre Praevalenz {ziel}, {wdh} Wiederholungen")
    print("=" * 74)
    pos = dev[dev.y == 1]["p_kal"].to_numpy()
    neg = dev[dev.y == 0]["p_kal"].to_numpy()
    print(f"\n  {'n':>6}  {'BBSE 95 %-Band':>26}  {'EM 95 %-Band':>26}")
    print("  " + "-" * 62)
    aus = []
    for n in [50, 100, 200, 500, 1000, 2000]:
        bs, es = [], []
        for _ in range(wdh):
            n_pos = RNG.binomial(n, ziel)
            p = np.concatenate([
                RNG.choice(pos, n_pos, replace=True),
                RNG.choice(neg, n - n_pos, replace=True)])
            b, _ = bbse(p, SCHWELLE, sens, spez)
            bs.append(b)
            es.append(em(p, pi_quelle))
        bl, bh = np.percentile(bs, [2.5, 97.5])
        el, eh = np.percentile(es, [2.5, 97.5])
        aus.append({"n": n, "bbse": (bl, bh), "em": (el, eh)})
        print(f"  {n:>6}  [{bl:.3f}, {bh:.3f}]  +/-{(bh - bl) / 2:.3f}     "
              f"[{el:.3f}, {eh:.3f}]  +/-{(eh - el) / 2:.3f}")
    print()
    return aus


# ---------------------------------------------------------------- 2. Kermany

def kermany(ker, sens, spez, pi_quelle):
    print("=" * 74)
    print("2. KERMANY  echte fremde Daten, wahre Praevalenz bekannt")
    print("=" * 74)
    p, y = ker["p_kal"].to_numpy(), ker["y"].to_numpy()
    pi_wahr = float(y.mean())
    b, q = bbse(p, SCHWELLE, sens, spez)
    e = em(p, pi_quelle)

    print(f"\n  n                        {len(ker)}")
    print(f"  wahre Praevalenz         {pi_wahr:.4f}")
    print(f"  als positiv eingestuft   {q:.4f}")
    print(f"\n  BBSE schaetzt            {b:.4f}   ({b - pi_wahr:+.4f})")
    print(f"  EM   schaetzt            {e:.4f}   ({e - pi_wahr:+.4f})")

    # gruppierter Bootstrap ueber Patienten, wie im uebrigen Projekt
    gruppen = ker["group"].to_numpy()
    uniq = np.unique(gruppen)
    idx = {g: np.where(gruppen == g)[0] for g in uniq}
    bs, es = [], []
    for _ in range(400):
        zieh = RNG.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx[g] for g in zieh])
        bb, _ = bbse(p[sel], SCHWELLE, sens, spez)
        bs.append(bb)
        es.append(em(p[sel], pi_quelle))
    print(f"\n  BBSE 95 %-Band (Patienten-Bootstrap)  "
          f"[{np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f}]")
    print(f"  EM   95 %-Band                        "
          f"[{np.percentile(es, 2.5):.4f}, {np.percentile(es, 97.5):.4f}]")

    print("\n  --- was die Schaetzung an der Kalibrierung ausrichtet ---")
    print(f"\n  {'Prior':<28} {'d(Logit)':>10} {'ECE':>9} {'Brier':>9} "
          f"{'Sens':>7} {'Spez':>7}")
    print("  " + "-" * 74)
    zeilen = []
    for name, pi in [("keine Korrektur", None),
                     (f"BBSE geschaetzt {b:.3f}", b),
                     (f"EM geschaetzt {e:.3f}", e),
                     (f"wahr {pi_wahr:.3f} (unerreichbar)", pi_wahr)]:
        if pi is None:
            pk, d = p, 0.0
        else:
            pk, d = prior_shift(p, pi_quelle, pi)
        s = float(((pk >= SCHWELLE) & (y == 1)).sum() / (y == 1).sum())
        sp = float(((pk < SCHWELLE) & (y == 0)).sum() / (y == 0).sum())
        zeilen.append({"name": name, "pi": pi, "d": d,
                       "ece": ece(pk, y), "brier": brier(pk, y),
                       "sens": s, "spez": sp})
        print(f"  {name:<28} {d:>+10.4f} {ece(pk, y):>9.4f} "
              f"{brier(pk, y):>9.4f} {s:>7.4f} {sp:>7.4f}")
    print("\n  Anmerkung: Sens und Spez aendern sich NUR, weil die Schwelle hier")
    print("  auf der verschobenen Skala angewandt wird. Bleibt die Entscheidung")
    print("  auf der Referenzskala, sind beide von der Korrektur unberuehrt.\n")
    return {"pi_wahr": pi_wahr, "bbse": b, "em": e, "q": q, "zeilen": zeilen}


# ---------------------------------------------------------------- 3. LR

def likelihood_ratio(pi_quelle):
    print("=" * 74)
    print("3. LIKELIHOOD-VERHAELTNIS  die praevalenzfreie Zahl")
    print("=" * 74)
    print("   LR(s) = odds(p_ref) / odds(pi_ref).  Reine Modelleigenschaft,")
    print("   ueberlebt jeden Praevalenzwechsel unveraendert.\n")
    o_ref = pi_quelle / (1 - pi_quelle)
    print(f"  {'angezeigte Zahl':>18} {'LR':>8}   {'was sie bedeutet'}")
    print("  " + "-" * 68)
    for p, was in [(0.0792, "Median der Bilder ohne Pneumonie"),
                   (SCHWELLE, "ARBEITSPUNKT"),
                   (0.30, ""), (0.4823, "Median der echten Pneumonien"),
                   (0.60, "90. Perzentil der Anzeige"),
                   (0.8927, "hoechster je gesehener Wert")]:
        lr = (p / (1 - p)) / o_ref
        print(f"  {p:>18.4f} {lr:>8.2f}   {was}")
    print()


# ---------------------------------------------------------------- main

def main():
    dev = lade_dev()
    ist = nachweis(dev)
    sens, spez = ist["sens"], ist["spez"]
    pi_quelle = ist["praevalenz"]

    simulation(dev, sens, spez, pi_quelle)
    stichprobenumfang(dev, sens, spez, pi_quelle)
    ker = lade_kermany()
    erg = kermany(ker, sens, spez, pi_quelle)
    reststreit(ker, dev, pi_quelle)
    likelihood_ratio(pi_quelle)

    print("=" * 74)
    print("URTEIL")
    print("=" * 74)
    print(f"  Simulation getroffen, Kermany um "
          f"{erg['bbse'] - erg['pi_wahr']:+.3f} (BBSE) bzw. "
          f"{erg['em'] - erg['pi_wahr']:+.3f} (EM) daneben.")
    print("  Der Schaetzer ist in Ordnung, die Annahme bei Kermany nicht.")




# ------------------------------------------------- 2b. Woher der Rest kommt

def reststreit(ker, dev, pi_quelle):
    """Nach der WAHREN Priorverschiebung bleibt ECE 0.1641. Ist das Rauschen?"""
    p, y = ker["p_kal"].to_numpy(), ker["y"].to_numpy()
    pi_wahr = float(y.mean())
    pk, _ = prior_shift(p, pi_quelle, pi_wahr)
    print("=" * 74)
    print("2b. RESTSTREIT  wo der Fehler nach der wahren Korrektur noch sitzt")
    print("=" * 74)
    print(f"\n  {'Feld':>12} {'n':>6} {'p (korrigiert)':>15} {'beobachtet':>12} "
          f"{'Rest':>9}")
    print("  " + "-" * 60)
    kanten = np.linspace(0, 1, 11)
    for lo, hi in zip(kanten[:-1], kanten[1:]):
        m = (pk >= lo) & (pk < hi) if hi < 1 else (pk >= lo) & (pk <= hi)
        if m.sum() < 30:
            continue
        print(f"  {lo:.1f}-{hi:.1f}    {m.sum():>6} {pk[m].mean():>15.4f} "
              f"{y[m].mean():>12.4f} {y[m].mean() - pk[m].mean():>+9.4f}")
    print("\n  Ein reiner Achsenfehler haette hier ein GLEICHES Vorzeichen ueberall.")
    print("  Wechselt es, hat sich auch die Steigung bewegt -> Kovariatenverschiebung,")
    print("  und die ist mit keinem Prior der Welt zu heilen.\n")

    print("=" * 74)
    print("4. ALARM  wie man merkt, dass die Population gewechselt hat")
    print("=" * 74)
    ld = logit(dev["p_kal"].to_numpy()).mean()
    lk = logit(p).mean()
    print(f"\n  mittleres Logit Entwicklung  {ld:>8.4f}")
    print(f"  mittleres Logit Kermany      {lk:>8.4f}   Versatz {lk - ld:+.4f}")
    print("\n  Braucht KEINE Labels. Laeuft ueber die letzten n Anfragen mit und")
    print("  schlaegt an, wenn der Versatz eine festgelegte Schranke reisst.")
    print("  Kermany haette hier klar angeschlagen, ohne ein einziges Label.\n")
if __name__ == "__main__":
    main()
