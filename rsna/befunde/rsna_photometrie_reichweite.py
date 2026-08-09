"""Phase 9, before the run: how far does the brightness knob actually reach?

WHY THIS SCRIPT EXISTS
----------------------
Phase 6 turned up the geometric augmentation and failed. The diagnosis came
AFTERWARDS: the knob had removed only 24 percent of the size cue, because the
scale range was one sided. A knob that cannot reach the confounder decides
nothing, and finding that out after eleven hours of training is the expensive
way to find it out.

For the photometric knob the same calculation can be done BEFORE the run, and
it can be done exactly. `ColorJitter(brightness=b)` multiplies every pixel by
one factor drawn per image from [1-b, 1+b], and `contrast=c` blends every pixel
towards the image mean by a factor from [1-c, 1+c]. Both are POINTWISE maps on
the grey value. A 256 bin histogram per image therefore carries everything
needed to compute the result exactly, including the clipping at 0 and 255,
which is where a large brightness factor loses its reach.

WHAT IS MEASURED
----------------
    AUC(image statistic -> ViewPosition)

for two statistics the jitter can touch: the mean grey value and its standard
deviation. The distance from the coin, |AUC - 0.5|, is what a stronger knob has
to shrink. The table reports how much of that distance SURVIVES at each
strength, in the same form phase 6 reported it afterwards.

WHAT THIS MEASURES AND WHAT IT DOES NOT
---------------------------------------
It measures what the knob can take away from the GLOBAL grey value. It does not
predict dC. Phase 8 showed why: when one channel is drained the network can
read the projection somewhere else, there from the fine texture that 512 pixels
had newly supplied. A knob with reach is a necessary condition for the arm to
work, not a sufficient one. That sentence belongs in the pre-registration and
not in the discussion afterwards.

Two further limits, both deliberate:

  * The histogram is taken from the resized image WITHOUT the affine
    augmentation. The affine runs before the jitter and its black fill lowers
    the mean a little. It is identical in both arms and small at scale 0.93 to
    1.07.
  * ImageNet normalisation runs AFTER the jitter and is a fixed affine map per
    channel. It cannot change an AUC, since AUC only reads the ordering.

THE DEVICE CHECK COMES FIRST
----------------------------
Standing rule of this project: a new measuring device is checked against a case
whose answer is known before it is used on the open question. Here the answer
is known twice over. At strength 0 the simulated AUC has to reproduce the
unjittered one exactly, and on a sample of real images the histogram simulation
has to reproduce what `PIL.ImageEnhance` actually produces, which is the code
path `torchvision.transforms.ColorJitter` takes for a PIL image.

USAGE
-----
    python rsna\\befunde\\rsna_photometrie_reichweite.py --histogramme
    python rsna\\befunde\\rsna_photometrie_reichweite.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

import _repo_path  # noqa: F401  (puts the neighbouring folders on sys.path)

# Die Staerken, die bewertet werden. 0 ist der Pflichtfall des Messgeraets,
# 0,15 ist der Ist-Zustand jedes Laufs dieses Projekts.
STAERKEN = [0.0, 0.15, 0.30, 0.40, 0.50, 0.60, 0.80]

WIEDERHOLUNGEN = 25       # Ziehungen je Staerke, fuer die Streuung der Simulation
SEED = 12345
STICHPROBE_GERAET = 300   # Bilder fuer die Geraetepruefung gegen PIL
TRAINING_PX = 224         # die Kantenlaenge, bei der der Arm laeuft

BEZUG_DIR = "predictions_final_model"   # liefert patientId, viewpos, y


# --------------------------------------------------------------------------
# Statistik
# --------------------------------------------------------------------------

def rank_auc(score, label) -> float:
    """AUC als Rangstatistik, Bindungen bekommen ihren mittleren Rang."""
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    if label.all() or not label.any():
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    n1 = int(label.sum())
    n0 = int((~label).sum())
    return float((r[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def abbruch(text: str) -> None:
    print(f"\nABBRUCH: {text}")
    sys.exit(1)


# --------------------------------------------------------------------------
# Der Jitter, einmal als Abbildung auf dem Grauwert
# --------------------------------------------------------------------------

WERTE = np.arange(256, dtype=np.float64)
BLOCK = 4000          # Bilder je Rechenblock, damit der Speicher reicht


def _blend(v: np.ndarray) -> np.ndarray:
    """Was PIL beim Ueberblenden mit einer Fliesskommazahl macht.

    NICHT runden, sondern ABSCHNEIDEN. `ImagingBlend` rechnet den Wert als
    float und castet ihn mit `(UINT8)temp`, und ein Cast in C schneidet ab.
    Wer hier rundet, verschiebt jeden Grauwert im Mittel um eine halbe Stufe,
    und der zweite der beiden Schritte multipliziert diesen Versatz noch mit
    seinem eigenen Faktor.

    GEFUNDEN VON DER GERAETEPRUEFUNG, nicht durch Lesen des Quelltextes: die
    erste Fassung rundete und lag im Bildmittel um bis zu 1,51 Grauwerte
    daneben. Genau dafuer steht die Pruefung vor der Tabelle.
    """
    return np.clip(np.floor(v), 0, 255)


def jitter_karten(hf: np.ndarray, fb: np.ndarray, fc: np.ndarray,
                  erst_helligkeit: np.ndarray) -> np.ndarray:
    """Die Abbildung Grauwert -> Grauwert, fuer viele Bilder auf einmal.

    PIL rechnet Helligkeit als Ueberblendung gegen Schwarz (v * f) und Kontrast
    als Ueberblendung gegen ein Vollbild im GERUNDETEN Bildmittel
    (m + f * (v - m)). Beides wird auf 0 bis 255 beschnitten, und genau dieses
    Beschneiden ist der Grund, warum ein grosser Helligkeitsfaktor an Reichweite
    verliert. Die Reihenfolge der beiden Schritte zieht ColorJitter selbst
    zufaellig, deshalb werden beide gerechnet und danach ausgewaehlt.

    hf ist (n, 256), fb / fc / erst_helligkeit sind (n,). Rueckgabe (n, 256).
    """
    w = WERTE[None, :]
    ntot = hf.sum(1, keepdims=True)
    # Reihenfolge A: erst Helligkeit, dann Kontrast um das NEUE Bildmittel
    a = _blend(w * fb[:, None])
    m = np.floor((hf * a).sum(1, keepdims=True) / ntot + 0.5)
    a = _blend(m + fc[:, None] * (a - m))
    # Reihenfolge B: erst Kontrast um das alte Bildmittel, dann Helligkeit
    m = np.floor((hf * w).sum(1, keepdims=True) / ntot + 0.5)
    b = _blend(m + fc[:, None] * (w - m))
    b = _blend(b * fb[:, None])
    return np.where(erst_helligkeit[:, None], a, b)


def statistiken(hf: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mittelwert und Streuung je Bild, nachdem die Abbildung angewandt wurde."""
    ntot = hf.sum(1)
    mu = (hf * k).sum(1) / ntot
    var = (hf * (k - mu[:, None]) ** 2).sum(1) / ntot
    return mu, np.sqrt(np.maximum(var, 0.0))


def gejittert(hf: np.ndarray, fb: np.ndarray, fc: np.ndarray,
              erst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dasselbe blockweise ueber alle Bilder, damit der Speicher reicht."""
    n = hf.shape[0]
    mu = np.empty(n)
    sd = np.empty(n)
    for i in range(0, n, BLOCK):
        j = min(i + BLOCK, n)
        k = jitter_karten(hf[i:j], fb[i:j], fc[i:j], erst[i:j])
        mu[i:j], sd[i:j] = statistiken(hf[i:j], k)
    return mu, sd


# --------------------------------------------------------------------------
# Stufe 1: die Histogramme
# --------------------------------------------------------------------------

def teilpfad(ziel: Path, von: int, bis: int) -> Path:
    return ziel.with_name(f"{ziel.stem}_teil_{von:05d}_{bis:05d}.npz")


def histogramme_bauen(bilder: Path, ids: np.ndarray, ziel: Path,
                      von: int, bis: int) -> None:
    """Stufe 1, abschnittsweise.

    Der Abschnitt ist kein Selbstzweck: 22872 Bilder brauchen einige Minuten,
    und ein Lauf, der nach der Haelfte abbricht, soll nicht von vorn anfangen
    muessen. Fertige Abschnitte werden uebersprungen, wie `run_phase8.ps1` es
    mit fertigen Folds haelt.
    """
    von = max(0, von)
    bis = len(ids) if bis <= 0 else min(bis, len(ids))
    p_teil = teilpfad(ziel, von, bis)
    if p_teil.exists():
        print(f"  {p_teil.name}: schon da, uebersprungen")
        return
    print(f"\n  Lese Bilder {von} bis {bis} aus {bilder}, verkleinert auf "
          f"{TRAINING_PX} px")
    h = np.zeros((bis - von, 256), dtype=np.int32)
    t0 = time.time()
    for k, pid in enumerate(ids[von:bis]):
        p = bilder / f"{pid}.png"
        if not p.exists():
            abbruch(f"{p} fehlt.")
        im = Image.open(p).convert("L").resize((TRAINING_PX, TRAINING_PX),
                                               Image.BILINEAR)
        h[k] = np.bincount(np.asarray(im).ravel(), minlength=256)
        if (k + 1) % 2000 == 0:
            print(f"    {k + 1} von {bis - von}  "
                  f"[{time.time() - t0:.0f} s]", flush=True)
    p_teil.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p_teil, ids=ids[von:bis], hist=h, px=TRAINING_PX,
                        von=von, bis=bis)
    print(f"  geschrieben: {p_teil}  ({time.time() - t0:.0f} s)")


def teile_zusammenfuegen(ids: np.ndarray, ziel: Path) -> None:
    """Die Abschnitte zu einer Datei, und dabei die Vollstaendigkeit pruefen.

    Geprueft wird nicht, ob Dateien da sind, sondern ob die zusammengesetzte
    ID-Liste Bild fuer Bild die aus dem Bezugsarm ist. Eine Luecke oder eine
    Dopplung faellt damit auf, eine fehlende Datei ebenfalls.
    """
    teile = sorted(ziel.parent.glob(f"{ziel.stem}_teil_*.npz"))
    if not teile:
        abbruch(f"keine Abschnitte in {ziel.parent} gefunden.")
    hs, id_teile = [], []
    for t in teile:
        z = np.load(t, allow_pickle=True)
        if int(z["px"]) != TRAINING_PX:
            abbruch(f"{t.name} wurde bei {int(z['px'])} px gebaut.")
        hs.append(z["hist"])
        id_teile.append(z["ids"])
    zusammen = np.concatenate(id_teile)
    if list(zusammen) != list(ids):
        abbruch(f"die {len(teile)} Abschnitte ergeben {len(zusammen)} Bilder "
                f"und nicht die {len(ids)} des Bezugsarms in dessen "
                f"Reihenfolge. Luecke, Dopplung oder fehlender Abschnitt.")
    np.savez_compressed(ziel, ids=ids, hist=np.concatenate(hs),
                        px=TRAINING_PX)
    print(f"  ok   {len(teile)} Abschnitte, {len(ids)} Bilder, Reihenfolge "
          f"stimmt Bild fuer Bild")
    print(f"  geschrieben: {ziel}")


# --------------------------------------------------------------------------
# Die Geraetepruefung
# --------------------------------------------------------------------------

def geraet_pruefen(bilder: Path, ids: np.ndarray, h: np.ndarray,
                   rng: np.random.Generator) -> None:
    """Die Simulation gegen das, was PIL wirklich rechnet.

    `torchvision.transforms.ColorJitter` ruft auf einem PIL-Bild
    `ImageEnhance.Brightness` und `ImageEnhance.Contrast` auf. Genau die werden
    hier auf echten Bildern ausgefuehrt und mit der Histogrammrechnung
    verglichen. Faellt das durch, ist jede Zahl weiter unten wertlos.
    """
    print("\n  GERAETEPRUEFUNG 1: die Simulation gegen PIL, auf echten Bildern")
    idx = rng.choice(len(ids), size=min(STICHPROBE_GERAET, len(ids)),
                     replace=False)
    ab_mu, ab_sd = 0.0, 0.0
    for i in idx:
        im = Image.open(bilder / f"{ids[i]}.png").convert("L").resize(
            (TRAINING_PX, TRAINING_PX), Image.BILINEAR)
        fb = float(rng.uniform(0.2, 1.8))
        fc = float(rng.uniform(0.2, 1.8))
        erst_h = bool(rng.integers(2))
        if erst_h:
            echt = ImageEnhance.Contrast(
                ImageEnhance.Brightness(im).enhance(fb)).enhance(fc)
        else:
            echt = ImageEnhance.Brightness(
                ImageEnhance.Contrast(im).enhance(fc)).enhance(fb)
        a = np.asarray(echt, dtype=np.float64)
        # GENAU die Funktion, die unten die Tabelle rechnet, mit n = 1. Eine
        # zweite Implementierung zu pruefen waere sinnlos.
        mu, sd = gejittert(h[i:i + 1], np.array([fb]), np.array([fc]),
                           np.array([erst_h]))
        ab_mu = max(ab_mu, abs(float(mu[0]) - a.mean()))
        ab_sd = max(ab_sd, abs(float(sd[0]) - a.std()))
    print(f"    groesste Abweichung im Mittelwert   {ab_mu:.4f} Grauwerte")
    print(f"    groesste Abweichung in der Streuung {ab_sd:.4f} Grauwerte")
    if ab_mu > 0.5 or ab_sd > 0.5:
        abbruch("die Histogrammrechnung gibt nicht wieder, was PIL tut. "
                "Ohne diese Uebereinstimmung ist die Reichweitentabelle "
                "Zahlenmalerei.")
    print("    ok   unter einem halben Grauwert, die Simulation bildet PIL ab")


# --------------------------------------------------------------------------
# Stufe 2: die Reichweite
# --------------------------------------------------------------------------

def reichweite(h: np.ndarray, ap: np.ndarray,
               rng: np.random.Generator) -> pd.DataFrame:
    hf = h.astype(np.float64)
    n = len(hf)
    eins = np.ones(n)

    # Ungestoert. Das ist der Bezug, gegen den alles gerechnet wird, und er
    # laeuft durch DIESELBE Funktion mit Faktoren von genau 1.
    mu0, sd0 = gejittert(hf, eins, eins, np.ones(n, dtype=bool))
    auc_mu0 = rank_auc(mu0, ap)
    auc_sd0 = rank_auc(sd0, ap)
    print(f"\n  ungestoert: AUC(Mittelwert -> Projektion) {auc_mu0:.4f}, "
          f"AUC(Streuung -> Projektion) {auc_sd0:.4f}")

    print("\n  GERAETEPRUEFUNG 2: bei Staerke 0 muss die Simulation die "
          "ungestoerte Zahl treffen")
    zeilen = []
    for b in STAERKEN:
        a_mu, a_sd = [], []
        for _ in range(WIEDERHOLUNGEN if b > 0 else 1):
            lo = max(0.0, 1.0 - b)
            fb = rng.uniform(lo, 1.0 + b, size=n)
            fc = rng.uniform(lo, 1.0 + b, size=n)
            erst = rng.integers(2, size=n).astype(bool)
            mu, sd = gejittert(hf, fb, fc, erst)
            a_mu.append(rank_auc(mu, ap))
            a_sd.append(rank_auc(sd, ap))
        zeilen.append({
            "b": b,
            "auc_mittel": float(np.mean(a_mu)),
            "sd_mittel": float(np.std(a_mu, ddof=1)) if len(a_mu) > 1 else 0.0,
            "auc_streuung": float(np.mean(a_sd)),
            "sd_streuung": float(np.std(a_sd, ddof=1)) if len(a_sd) > 1 else 0.0,
        })
        if b == 0.0:
            d_mu = abs(zeilen[-1]["auc_mittel"] - auc_mu0)
            d_sd = abs(zeilen[-1]["auc_streuung"] - auc_sd0)
            print(f"    Mittelwert  {d_mu:.2e}     Streuung  {d_sd:.2e}")
            if max(d_mu, d_sd) > 1e-12:
                abbruch("bei Staerke 0 zieht die Simulation Faktoren, die "
                        "nicht genau 1 sind, oder rundet anders. Dann ist "
                        "jede Zeile darunter verschoben.")
            print("    ok   exakt getroffen")
        else:
            print(f"      b = {b:.2f} gerechnet", flush=True)

    d = pd.DataFrame(zeilen)
    d["uebrig_mittel"] = (d["auc_mittel"] - 0.5).abs() / abs(auc_mu0 - 0.5)
    d["uebrig_streuung"] = (d["auc_streuung"] - 0.5).abs() / abs(auc_sd0 - 0.5)
    return d


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bilder", type=Path, default=Path("data/rsna/png512"))
    p.add_argument("--bezug-dir", type=Path, default=Path(BEZUG_DIR))
    p.add_argument("--cache", type=Path,
                   default=Path("qc/photometrie_histogramme.npz"))
    p.add_argument("--histogramme", action="store_true",
                   help="Stufe 1: Histogramme eines Abschnitts bauen")
    p.add_argument("--von", type=int, default=0)
    p.add_argument("--bis", type=int, default=0, help="0 heisst bis zum Ende")
    p.add_argument("--zusammenfuegen", action="store_true",
                   help="Stufe 1b: die Abschnitte zu einer Datei")
    args = p.parse_args()

    # Die Bildliste kommt aus den Vorhersagedateien des BEZUGSARMS und nicht
    # aus einer neuen Quelle. Damit ist die Menge dieselbe, auf der die Tore
    # spaeter gelesen werden, und die Projektion kommt aus derselben Spalte.
    teile = []
    for f in range(5):
        q = args.bezug_dir / f"rsna_f{f}_s0.csv"
        if not q.exists():
            abbruch(f"{q} fehlt. Die Bildliste stammt aus dem Bezugsarm.")
        teile.append(pd.read_csv(q, usecols=["patientId", "viewpos", "y"]))
    d = pd.concat(teile).drop_duplicates(subset="patientId")
    d = d.sort_values("patientId").reset_index(drop=True)
    ids = d["patientId"].to_numpy()
    ap = (d["viewpos"] == "AP").to_numpy()
    print("=" * 78)
    print("REICHWEITE DES PHOTOMETRISCHEN KNOPFS, vor dem Lauf")
    print("=" * 78)
    print(f"  {len(ids)} Bilder aus {args.bezug_dir}, davon {int(ap.sum())} AP "
          f"und {int((~ap).sum())} PA")

    if args.histogramme:
        histogramme_bauen(args.bilder, ids, args.cache, args.von, args.bis)
        return

    if args.zusammenfuegen:
        teile_zusammenfuegen(ids, args.cache)
        return

    if not args.cache.exists():
        abbruch(f"{args.cache} fehlt. Erst Stufe 1 laufen lassen:\n"
                f"  python {Path(__file__).name} --histogramme")
    z = np.load(args.cache, allow_pickle=True)
    if list(z["ids"]) != list(ids):
        abbruch("die Bildliste im Zwischenspeicher ist nicht die aus dem "
                "Bezugsarm. Stufe 1 neu rechnen.")
    if int(z["px"]) != TRAINING_PX:
        abbruch(f"der Zwischenspeicher wurde bei {int(z['px'])} px gebaut, "
                f"gebraucht werden {TRAINING_PX}.")
    h = z["hist"]

    rng = np.random.default_rng(SEED)
    geraet_pruefen(args.bilder, ids, h.astype(np.float64), rng)
    d = reichweite(h, ap, rng)

    print("\n" + "=" * 78)
    print("DIE REICHWEITE")
    print("=" * 78)
    print("  b ist die Staerke von brightness UND contrast zugleich, also der")
    print("  Faktorbereich [1-b, 1+b]. 'uebrig' ist der Anteil des Abstands zur")
    print("  Muenze, der die Stoerung ueberlebt. Kleiner ist besser.")
    print()
    print(f"  {'b':>6}{'AUC Mittel':>13}{'uebrig':>9}"
          f"{'AUC Streuung':>15}{'uebrig':>9}")
    for _, r in d.iterrows():
        print(f"  {r['b']:>6.2f}{r['auc_mittel']:>13.4f}"
              f"{r['uebrig_mittel']:>8.0%}"
              f"{r['auc_streuung']:>15.4f}{r['uebrig_streuung']:>8.0%}")
    print()
    print("  Streuung der Simulation ueber die Ziehungen, groesster Wert: "
          f"{max(d['sd_mittel'].max(), d['sd_streuung'].max()):.4f} AUC.")
    print()
    print("  ZUR EINORDNUNG: Phase 6 nahm dem Groessenhinweis 24 Prozent weg")
    print("  und bewegte C um 0,005. Ein Knopf, der hier weniger als die")
    print("  Haelfte wegnimmt, waere derselbe Fehler noch einmal.")
    print()
    print("  UND DIE GRENZE, die auch nach dem Lauf gilt: das hier ist der")
    print("  GLOBALE Grauwert. Phase 8 hat gezeigt, dass der Kanal umziehen")
    print("  kann, wenn man ihm an einer Stelle das Wasser abgraebt. Reichweite")
    print("  ist notwendig, nicht hinreichend.")


if __name__ == "__main__":
    main()
