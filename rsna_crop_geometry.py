"""
Schritt 9d: Greift ein Rechteck-Crop den ViewPosition-Confounder an?

Die Frage
---------
Mikas Vorschlag: nicht pixelgenau auf die Lungenmaske zuschneiden, sondern das
UMSCHLIESSENDE RECHTECK plus Rand nehmen. Robuster gegen Segmentierungsfehler,
weil das Rechteck von Lungenspitzen und Zwerchfellwinkeln bestimmt wird -- und
die segmentiert das U-Net zuverlaessig, auch wenn es eine Konsolidierung in der
Mitte verliert.

Der naheliegende Wirkmechanismus waere die effektive Aufloesung. Der ist
allerdings schon eingegrenzt: der Crop braechte 1,1x bis 1,4x linearen Zoom,
und das Modell verliert bei 0,45x Aufloesung nur 0,016 AUC. Erwartbarer Effekt
also rund 0,005 -- unter der Fold-Streuung von 0,015.

Es gibt aber einen zweiten, staerkeren Mechanismus, und den prueft dieses
Skript: **der Crop normalisiert den Bildausschnitt**. AP-Liegendaufnahmen sind
anders gerahmt als PA-Stehendaufnahmen -- mehr Schulter, mehr Abdomen, anderer
Abstand, andere Zentrierung. Wenn das so ist, traegt die Rahmung Information
ueber die Projektion, und die Projektion ist mit +0,044 die groesste bekannte
Stoergroesse (ViewPosition -> Target: AUC 0,706; der Modellscore sagt die
Projektion mit AUC 0,808 vorher, liest sie also aus dem Bild).

Ein Crop, der die Rahmung wegnormiert, wuerde diesen Kanal schliessen. Das
waere ein viel besserer Grund als Aufloesung.

Zwei Messungen, beide noetig
----------------------------
1. **AUC(Rechteck-Geometrie -> ViewPosition).** Hoch = die Rahmung ist ein
   Projektionsproxy, der Crop hat etwas zu normalisieren. Nahe 0,5 = es gibt
   nichts wegzunehmen, und nur der schwache Aufloesungsgrund bleibt.

2. **AUC(Rechteck-Geometrie -> Target).** Die Gegenprobe. Wenn die
   Crop-Parameter selbst die Klasse verraten, baut ein Zuschnitt nach diesen
   Parametern einen NEUEN Shortcut ein -- genau der Fehler, der auf Kermany
   passiert ist (Maskenflaeche AUC 0,255). Diese Zahl muss nahe 0,5 liegen,
   sonst ist der Crop schaedlich, egal was Messung 1 sagt.

Eine Designfolge steht schon fest
---------------------------------
Ein nicht-quadratisches Rechteck auf 224x224 zu skalieren verzerrt das Bild,
und zwar je nach Seitenverhaeltnis verschieden stark. Das Seitenverhaeltnis
ueberlebt den Crop dann als Verzerrungssignatur -- der Kanal waere nicht
geschlossen, sondern nur umkodiert. Deshalb wird `aspect` hier einzeln
ausgewiesen: ist es allein schon ein guter Projektionsproxy, muss der Crop auf
ein QUADRATISCHES Rechteck erweitert werden (kuerzere Seite aufziehen).

CLI:
  python rsna_crop_geometry.py                       # nutzt den Roh-Cache
  python rsna_crop_geometry.py --refine hull --dilate-px 8 --pad 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_make_masks import refine_variant, unpack_masks

FEATURES = ["y0", "y1", "x0", "x1", "height", "width", "aspect", "cy", "cx",
            "area"]


def rect_features(mask: np.ndarray, pad: float) -> dict | None:
    """Die Parameter, nach denen zugeschnitten wuerde -- als Merkmalsvektor.

    Genau diese Groessen bestimmen den Zuschnitt. Wenn ein Klassifikator aus
    ihnen die Projektion (oder schlimmer: die Klasse) vorhersagen kann, dann
    traegt der Zuschnitt selbst diese Information.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    H, W = mask.shape
    ph = pad * (ys.max() - ys.min() + 1)
    pw = pad * (xs.max() - xs.min() + 1)
    y0, y1 = max(ys.min() - ph, 0), min(ys.max() + ph, H - 1)
    x0, x1 = max(xs.min() - pw, 0), min(xs.max() + pw, W - 1)
    h, w = y1 - y0 + 1, x1 - x0 + 1
    return {"y0": y0 / H, "y1": y1 / H, "x0": x0 / W, "x1": x1 / W,
            "height": h / H, "width": w / W, "aspect": w / h,
            "cy": (y0 + y1) / 2 / H, "cx": (x0 + x1) / 2 / W,
            "area": (h * w) / (H * W)}


def cv_auc(X: np.ndarray, y: np.ndarray, seed: int = 0, folds: int = 5) -> float:
    """AUC eines logistischen Modells in gruppenfreier CV.

    CV und nicht Anpassung auf denselben Daten: zehn Merkmale auf 1500 Punkten
    wuerden in-sample eine schmeichelhafte Zahl liefern. Jede RSNA-patientId
    kommt genau einmal vor, deshalb reicht hier eine geschichtete Aufteilung --
    es gibt keine Gruppen, die auseinandergerissen werden koennten.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(set(y)) < 2:
        return float("nan")
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def stratified_auc(df: pd.DataFrame, target: np.ndarray,
                   strata: np.ndarray) -> tuple[float, dict]:
    """AUC innerhalb jeder Projektion, dann nach Groesse gewichtet gemittelt.

    Warum das hier zwingend ist: `ViewPosition -> Target` hat AUC 0,706 und
    `Geometrie -> ViewPosition` 0,745. Die Geometrie kann die Klasse also
    vollstaendig DURCH die Projektion vorhersagen, ohne einen einzigen eigenen
    Beitrag. Eine ungeschichtete Zahl kann zwischen "neuer Shortcut" und
    "bekannter Confounder leitet durch" nicht unterscheiden -- und nur der
    erste Fall spraeche gegen den Zuschnitt.

    (Die erste Fassung dieses Skripts hat genau diesen Fehler gemacht und bei
    0,621 gewarnt. Geschichtet bleiben davon 0,541.)
    """
    per, tot, n = {}, 0.0, 0
    for s in sorted(set(strata)):
        sel = strata == s
        if len(set(target[sel])) < 2 or sel.sum() < 50:
            continue
        a = cv_auc(df[FEATURES].values[sel], target[sel])
        per[s] = (a, int(sel.sum()))
        tot += a * sel.sum()
        n += int(sel.sum())
    return (tot / n if n else float("nan")), per


def report_block(name: str, df: pd.DataFrame, target: np.ndarray,
                 refs: list[tuple[str, float]]) -> float:
    print(f"\n  {name}   (n = {len(df)}, positiv = {int(target.sum())})")
    full = cv_auc(df[FEATURES].values, target)
    print(f"    alle 10 Geometriemerkmale        AUC {full:.3f}")
    for f in ("aspect", "area", "height", "cy"):
        a = cv_auc(df[[f]].values, target)
        print(f"    nur {f:<28} AUC {a:.3f}")
    for label, val in refs:
        print(f"    (Vergleich: {label} {val:.3f})")
    return full


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-cache", type=Path, default=Path("data/rsna/unet_raw256.npz"))
    p.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    p.add_argument("--refine", default="hull")
    p.add_argument("--dilate-px", type=int, default=8)
    p.add_argument("--pad", type=float, default=0.05)
    p.add_argument("--out", type=Path,
                   default=Path("predictions_rsna/crop_geometry.csv"))
    args = p.parse_args(argv)

    if not args.raw_cache.exists():
        print(f"FEHLER: Roh-Cache fehlt: {args.raw_cache}")
        return 2

    sp = json.loads(args.splits.read_text())
    labels = {k: int(v) for k, v in sp["labels"].items()}
    vpmap = sp["viewpos"]

    z = np.load(args.raw_cache, allow_pickle=False)
    ids = [str(s) for s in z["ids"]]
    packed = z["packed"]
    print(f"Roh-Cache: {len(ids)} Masken | refine={args.refine} "
          f"dilate={args.dilate_px} pad={args.pad}")

    rows = []
    for i, pid in enumerate(ids):
        m = refine_variant(unpack_masks(packed[i:i + 1])[0], args.refine,
                           args.dilate_px)
        f = rect_features(m, args.pad)
        if f is None:
            continue
        f["patientId"] = pid
        f["target"] = labels.get(pid, -1)
        f["vp"] = vpmap.get(pid, "?")
        rows.append(f)
    d = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out, index=False)

    print(f"\nZuschnitt-Rechteck: Flaeche {d.area.mean():.3f} +- {d.area.std():.3f}"
          f" | linearer Zoom {(1 / np.sqrt(d.area)).mean():.2f}x")
    print(f"Seitenverhaeltnis (Breite/Hoehe): {d.aspect.mean():.3f} "
          f"+- {d.aspect.std():.3f}")

    print("\n" + "=" * 70)
    print("MESSUNG 1 -- traegt die Rahmung Information ueber die Projektion?")
    print("=" * 70)
    dv = d[d.vp.isin(["AP", "PA"])]
    vp_auc = report_block("Geometrie -> ViewPosition (AP=1)", dv,
                          (dv.vp == "AP").astype(int).values,
                          [("Modellscore -> ViewPosition", 0.808)])

    print("\n" + "=" * 70)
    print("MESSUNG 2 -- verraet der Zuschnitt selbst die Klasse?")
    print("=" * 70)
    dt = d[d.target >= 0]
    if dt.target.nunique() < 2:
        t_auc = float("nan")
        print(f"\n  NICHT BESTIMMBAR: der Cache enthaelt nur Target="
              f"{int(dt.target.iloc[0])}.")
        print("  Grad-CAM wurde nur auf positiven Bildern gemessen, deshalb sind")
        print("  im Roh-Cache keine Negativen. Fuer diese Zahl fehlen Masken:")
        print("    python rsna_make_masks.py --ids-from <CSV mit Negativen> \\")
        print("        --raw-cache data/rsna/unet_raw256.npz")
    else:
        report_block("Geometrie -> Target, UNGESCHICHTET", dt, dt.target.values,
                     [("ViewPosition allein -> Target", 0.706),
                      ("Header-Baseline", 0.557)])

        # Die ungeschichtete Zahl ist nicht auswertbar (siehe stratified_auc).
        # Massgeblich ist ausschliesslich der Rest innerhalb einer Projektion.
        ds = dt[dt.vp.isin(["AP", "PA"])]
        t_auc, per = stratified_auc(ds, ds.target.values, ds.vp.values)
        print("\n  Geschichtet nach ViewPosition -- traegt die Geometrie EIGENES Signal?")
        for s, (a, n) in per.items():
            print(f"    nur {s:<28} AUC {a:.3f}   (n = {n})")
        print(f"    gewichtetes Mittel               AUC {t_auc:.3f}"
              "   <- die massgebliche Zahl")

    print("\n" + "=" * 70)
    print("LESART")
    print("=" * 70)
    asp_vp = cv_auc(dv[["aspect"]].values, (dv.vp == "AP").astype(int).values)
    if vp_auc >= 0.70:
        print(f"  Die Rahmung sagt die Projektion mit AUC {vp_auc:.3f} vorher --")
        print("  es gibt also etwas wegzunormieren. ABER zwei Einschraenkungen:")
        print(f"  1. Das Seitenverhaeltnis allein holt davon schon {asp_vp:.3f}, und")
        print("     das ist Anatomie: es bleibt in den Pixeln sichtbar, egal wie")
        print(f"     zugeschnitten wird. Entfernbar ist nur der Rest (~{vp_auc - asp_vp:+.3f}).")
        print("  2. Die berichtete Zahl ist ohnehin GESCHICHTET (0,845). Der Crop")
        print("     wuerde sie kaum bewegen. Was er aendern koennte, ist die")
        print("     ABHAENGIGKEIT des Modells von der Projektion -- messbar daran,")
        print("     ob 'Modellscore -> ViewPosition' von 0,808 faellt. Das ist ein")
        print("     legitimes Ziel, aber eine andere Behauptung als 'bessere AUC'.")
    elif vp_auc >= 0.60:
        print(f"  Mittel (AUC {vp_auc:.3f}): die Rahmung traegt etwas, aber wenig.")
        print("  Der Crop wuerde einen Teilkanal schliessen. Grenzfall.")
    else:
        print(f"  Die Rahmung sagt die Projektion kaum vorher (AUC {vp_auc:.3f}).")
        print("  Es gibt nichts wegzunormieren -- der Crop wirkt dann nur ueber die")
        print("  Aufloesung, und die ist auf ~0,005 eingegrenzt.")
        print("  -> Kein hinreichender Grund fuer 2,3 h je Fold.")

    if np.isnan(t_auc):
        pass
    elif abs(t_auc - 0.5) > 0.08:
        print(f"\n  WARNUNG: die Zuschnitt-Parameter verraten die Klasse auch")
        print(f"  INNERHALB einer Projektion (AUC {t_auc:.3f}). Das ist ein neuer,")
        print("  eigenstaendiger Shortcut -- derselbe Fehler wie die Maskenflaeche")
        print("  auf Kermany (AUC 0,255). Zuschnitt so nicht verwendbar.")
    else:
        print(f"\n  Entwarnung beim Shortcut: geschichtet bleiben nur {t_auc:.3f}.")
        print("  Der ungeschichtete Wert entsteht fast vollstaendig dadurch, dass")
        print("  die Geometrie die PROJEKTION vorhersagt und die Projektion die")
        print("  Klasse. Kein neuer Kanal, sondern der bekannte, der durchleitet.")

    asp = asp_vp
    if asp >= 0.60:
        print(f"\n  DESIGNFOLGE: das Seitenverhaeltnis allein sagt die Projektion")
        print(f"  mit AUC {asp:.3f} vorher. Ein nicht-quadratischer Zuschnitt, auf")
        print("  224x224 gestreckt, kodiert diesen Kanal als Verzerrung neu statt")
        print("  ihn zu schliessen. Also auf ein QUADRATISCHES Rechteck erweitern")
        print("  (kuerzere Seite aufziehen), nicht die Seiten einzeln skalieren.")
    print("=" * 70)
    print(f"\nMerkmale: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
