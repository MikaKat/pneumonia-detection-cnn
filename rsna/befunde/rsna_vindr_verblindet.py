"""A blinded re-read: are the single-opinion VinDr positives really positive?

The question
------------
Of the 1588 VinDr images carrying an opacity box, 750 rest on ONE radiologist
whom two colleagues contradicted. On exactly those the localisation head
collapses to 0.6691, against 0.8493 where all three agreed. Two readings of the
same fact are possible and the numbers cannot separate them:

  (a) the label is noise, one reader over-called, the model is right to
      point elsewhere, or
  (b) the finding is subtle, two readers missed it, and the model missed it too.

Mika looked at ten of them and judged (a). That is expert evidence and it is
also an unblinded impression formed after seeing the result, on a sample that
was sorted by the model's own performance. This script turns it into a
measurement instead.

HOW A BLIND READ ANSWERS IT, AND WHY THE CONTROLS ARE THE WHOLE TRICK
---------------------------------------------------------------------
Reading the 750 disputed images alone would prove nothing: a rate of "40 %
positive" means nothing without knowing how this reader compares with VinDr's
panel in the first place. So the set is three strata, shuffled together and
indistinguishable:

  30 disputed        exactly one of three radiologists drew an opacity box
  15 consensus       at least two agreed          -> positive control
  15 clean negative  no radiologist saw anything  -> negative control

The two controls measure the reader against the panel. Only then does the rate
on the disputed images mean something.

THE READING RULE, FIXED BEFORE THE FIRST IMAGE IS OPENED
---------------------------------------------------------
Let p_neg be the reader's positive rate on the clean negatives and p_pos the
rate on the consensus positives. Both are properties of the reader, not of the
model. Then, for the disputed rate p_dis:

  p_dis below (p_neg + p_pos) / 2   ->  reading (a): the label is noise
  p_dis above it                    ->  reading (b): the finding is real
                                        and two readers missed it

Declared here, before any reading, so that neither outcome can be talked into
the other afterwards. And a warning that belongs here rather than after: with
30 disputed images a rate near 50 % carries a 95 % interval of roughly plus or
minus 18 points. This settles a direction, not a decimal.

WHAT THE READER DOES NOT SEE
-----------------------------
No boxes, no head field, no model score, no group, no file name that betrays
anything. The images are plain copies in shuffled order. The key is written to
a separate file that the evaluation reads and the reader does not.

One accident helps here: the 512 px release is already squashed to a square, so
the original image dimensions are gone from the pixels. Those dimensions
separate the classes at AUC 0.78 in this dataset, and in a full-resolution read
they would be an unconscious cue. Here they cannot be.

LICENCE
-------
VinDr images stay on this machine, exactly as in `rsna_vindr_einzelmeinung.py`.
The output folder is written into `.gitignore` by this script. The result may be
reported, the pictures may not.

  # Mappe bauen
  venv\\Scripts\\python.exe rsna\\befunde\\rsna_vindr_verblindet.py bauen

  # ... befundung.csv ausfuellen: 0 = keine Verschattung, 1 = fraglich,
  #     2 = Verschattung. Den Schluessel dabei NICHT oeffnen.

  venv\\Scripts\\python.exe rsna\\befunde\\rsna_vindr_verblindet.py auswerten
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260813
N_STRITTIG, N_KONSENS, N_NEGATIV = 30, 15, 15
URTEILE = {0: "keine Verschattung", 1: "fraglich", 2: "Verschattung"}


def gitignore_sichern(ordner: Path) -> None:
    gi = Path(".gitignore")
    eintrag = f"{ordner.as_posix()}/"
    if gi.is_file() and eintrag in gi.read_text(encoding="utf-8"):
        print(f"  .gitignore kennt {eintrag} bereits.")
        return
    with open(gi, "a", encoding="utf-8") as f:
        f.write(f"\n# VinDr-Bilder duerfen nicht weitergegeben werden "
                f"(PhysioNet Credentialed 1.5.0)\n{eintrag}\n")
    print(f"  {eintrag} in .gitignore eingetragen.")


def bauen(a) -> None:
    d = pd.read_csv(a.ens)
    if "n_rad" not in d.columns:
        raise SystemExit(f"{a.ens} hat keine Spalte n_rad. Erst "
                         f"rsna_extern_vindr_ens.py --lesart A laufen lassen.")
    rng = np.random.default_rng(SEED)

    gruppen = {"strittig": d[d.n_rad == 1], "konsens": d[d.n_rad >= 2],
               "negativ": d[d.n_rad == 0]}
    wunsch = {"strittig": N_STRITTIG, "konsens": N_KONSENS, "negativ": N_NEGATIV}
    gewaehlt = []
    for name, g in gruppen.items():
        n = min(wunsch[name], len(g))
        if n < wunsch[name]:
            print(f"  ACHTUNG: nur {n} statt {wunsch[name]} in '{name}'")
        idx = rng.choice(len(g), n, replace=False)
        teil = g.iloc[idx][["image_id", "n_rad", "p_ens"]].copy()
        teil["gruppe"] = name
        gewaehlt.append(teil)

    # Rein zufaellig gezogen, OHNE Blick auf den Modellwert. Waere nach dem
    # Score geschichtet worden, koennte die Nachbefundung das Modell nur noch
    # bestaetigen, statt es zu pruefen.
    mappe = pd.concat(gewaehlt, ignore_index=True)
    mappe = mappe.iloc[rng.permutation(len(mappe))].reset_index(drop=True)
    mappe.insert(0, "fall", [f"{i:02d}" for i in range(1, len(mappe) + 1)])

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "bilder").mkdir(exist_ok=True)
    gitignore_sichern(a.out)

    for r in mappe.itertuples():
        quelle = a.bilder / f"{r.image_id}.png"
        if not quelle.is_file():
            raise SystemExit(f"{quelle} fehlt")
        shutil.copyfile(quelle, a.out / "bilder" / f"fall_{r.fall}.png")

    # Der Bogen: nur Fallnummer und eine leere Spalte. Nichts sonst.
    pd.DataFrame({"fall": mappe.fall, "urteil": ""}).to_csv(
        a.out / "befundung.csv", index=False)
    # Der Schluessel, getrennt und mit einem Namen, der sich selbst erklaert.
    mappe.to_csv(a.out / "SCHLUESSEL_NICHT_VORHER_OEFFNEN.csv", index=False)

    (a.out / "ANLEITUNG.txt").write_text(
        "VERBLINDETE NACHBEFUNDUNG\n"
        "=========================\n\n"
        f"{len(mappe)} Thoraxaufnahmen in bilder/, in zufaelliger Reihenfolge.\n"
        "Frage je Bild: ist eine pneumonieverdaechtige VERSCHATTUNG zu sehen?\n\n"
        "In befundung.csv die Spalte 'urteil' ausfuellen:\n"
        "    0 = keine Verschattung\n"
        "    1 = fraglich\n"
        "    2 = Verschattung\n\n"
        "Die Datei SCHLUESSEL_NICHT_VORHER_OEFFNEN.csv bitte erst danach.\n"
        "Sie enthaelt, aus welcher Gruppe jeder Fall stammt, und ein Blick\n"
        "hinein macht die Messung wertlos.\n\n"
        "Danach:\n"
        "  venv\\Scripts\\python.exe rsna\\befunde\\rsna_vindr_verblindet.py auswerten\n",
        encoding="utf-8")

    print(f"\n  {len(mappe)} Faelle in {a.out / 'bilder'}")
    print(f"  Bogen:      {a.out / 'befundung.csv'}")
    print(f"  Schluessel: {a.out / 'SCHLUESSEL_NICHT_VORHER_OEFFNEN.csv'}  "
          f"(erst hinterher)")
    print("\n  Zusammensetzung steht im Schluessel und wird hier bewusst")
    print("  nicht gedruckt, damit der Bildschirm sie nicht verraet.")


def anteil_ci(k: int, n: int):
    """Wilson-Intervall. Bei n = 15 ist Wald sinnlos, das Band reicht dort
    sonst ueber die Null hinaus."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p, z = k / n, 1.959964
    nenner = 1 + z * z / n
    mitte = (p + z * z / (2 * n)) / nenner
    halb = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / nenner
    return p, max(mitte - halb, 0.0), min(mitte + halb, 1.0)


def auswerten(a) -> None:
    # Excel-feste Fassung. Zwei Fallen, beide auf Windows schon dagewesen:
    # es macht aus "01" die Zahl 1, und es schreibt leere Zellen mal als NaN
    # und mal als Leerstring. Beides wuerde hier still danebengehen: ein
    # kaputtes `merge` liefert eine LEERE Tabelle und keinen Fehler.
    def lies(pfad):
        t = pd.read_csv(pfad, dtype=str)
        t["fall"] = t.fall.astype(str).str.strip().str.zfill(2)
        return t

    bogen, schl = lies(a.out / "befundung.csv"), \
        lies(a.out / "SCHLUESSEL_NICHT_VORHER_OEFFNEN.csv")
    roh = bogen.urteil.fillna("").astype(str).str.strip()
    if (roh == "").any():
        raise SystemExit(f"{int((roh == '').sum())} von {len(roh)} Faellen sind "
                         f"noch nicht befundet.")
    if not roh.isin(["0", "1", "2"]).all():
        schlecht = sorted(set(roh) - {"0", "1", "2"})
        raise SystemExit(f"urteil darf nur 0, 1 oder 2 sein, gefunden: {schlecht}")
    bogen["urteil"] = roh.astype(int)

    m = schl.merge(bogen, on="fall")
    if len(m) != len(schl):
        raise SystemExit(f"nur {len(m)} von {len(schl)} Faellen liessen sich "
                         f"zuordnen. Wurde der Bogen umsortiert oder die "
                         f"Fallnummer veraendert?")
    m["p_ens"] = m.p_ens.astype(float)

    print("=" * 74)
    print("VERBLINDETE NACHBEFUNDUNG: das Ergebnis")
    print("=" * 74)
    print("\n  Verteilung der Urteile je Gruppe:\n")
    print(f"  {'Gruppe':<12}{'n':>4}" + "".join(f"{URTEILE[u]:>20}" for u in (0, 1, 2)))
    print("  " + "-" * 76)
    for g in ["negativ", "strittig", "konsens"]:
        s = m[m.gruppe == g]
        print(f"  {g:<12}{len(s):>4}" +
              "".join(f"{(s.urteil == u).sum():>10}"
                      f"{(s.urteil == u).mean():>9.0%}" for u in (0, 1, 2)))

    for streng, titel in [(True, "STRENG: nur Urteil 2 zaehlt als positiv"),
                          (False, "MILD: Urteil 1 oder 2 zaehlt als positiv")]:
        m["pos"] = (m.urteil == 2) if streng else (m.urteil >= 1)
        raten = {}
        print(f"\n  --- {titel} ---\n")
        print(f"  {'Gruppe':<12}{'positiv genannt':>18}{'95 %-Band':>22}")
        for g in ["negativ", "strittig", "konsens"]:
            s = m[m.gruppe == g]
            p, lo, hi = anteil_ci(int(s.pos.sum()), len(s))
            raten[g] = p
            print(f"  {g:<12}{int(s.pos.sum()):>6} von {len(s):<3} = {p:>5.0%}"
                  f"   [{lo:.0%}, {hi:.0%}]")
        mitte = (raten["negativ"] + raten["konsens"]) / 2
        print(f"\n  Mittelpunkt zwischen den Kontrollen: {mitte:.0%}")
        print(f"  Rate auf den strittigen Bildern:     {raten['strittig']:.0%}")
        if raten["konsens"] - raten["negativ"] < 0.25:
            print("\n  ACHTUNG: die beiden Kontrollen liegen weniger als 25 Punkte")
            print("  auseinander. Dann trennt die Nachbefundung selbst zu wenig,")
            print("  und die Zahl auf den Strittigen ist nicht deutbar.")
        elif raten["strittig"] < mitte:
            print("\n  -> LESART (a): die strittigen Bilder liegen naeher an den")
            print("     sicheren Negativen. Das VinDr-Label ist auf ihnen")
            print("     verrauscht, und der Einbruch des Kopffelds dort ist zu")
            print("     einem Teil dem Label anzulasten, nicht dem Modell.")
        else:
            print("\n  -> LESART (b): die strittigen Bilder liegen naeher an den")
            print("     Konsensbefunden. Dann ist dort meist etwas zu sehen, zwei")
            print("     Radiologen haben es uebersehen, und das Modell auch.")

    # Nebenrechnung, ausdruecklich als solche: was sagt das Modell dazu?
    m["pos"] = m.urteil >= 1
    print("\n" + "=" * 74)
    print("NEBENRECHNUNG  Modellwert gegen das neue Urteil, nur auf den Strittigen")
    print("=" * 74)
    s = m[m.gruppe == "strittig"]
    for lab, sub in [("als Verschattung befundet", s[s.urteil == 2]),
                     ("fraglich", s[s.urteil == 1]),
                     ("keine Verschattung", s[s.urteil == 0])]:
        if len(sub):
            print(f"  {lab:<28} n {len(sub):>3}   Median p {sub.p_ens.median():.4f}")
    print("\n  Nicht vorfestgelegt, also eine Beobachtung und kein Ergebnis.")

    aus = {"n": int(len(m)), "seed": SEED,
           "je_gruppe": {g: {"n": int((m.gruppe == g).sum()),
                             "urteile": {str(u): int(((m.gruppe == g) &
                                                      (m.urteil == u)).sum())
                                         for u in (0, 1, 2)}}
                         for g in ["negativ", "strittig", "konsens"]}}
    (a.out / "ergebnis.json").write_text(json.dumps(aus, indent=2))
    print(f"\n  -> {a.out / 'ergebnis.json'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("modus", choices=["bauen", "auswerten"])
    ap.add_argument("--bilder", type=Path,
                    default=Path("data/vinbigdata/vinbigdata/train"))
    ap.add_argument("--ens", type=Path,
                    default=Path("predictions_extern_vindr/extern_vindr_ens_A.csv"))
    ap.add_argument("--out", type=Path, default=Path("qc/vindr_verblindet"))
    a = ap.parse_args(argv)
    (bauen if a.modus == "bauen" else auswerten)(a)


if __name__ == "__main__":
    main()
