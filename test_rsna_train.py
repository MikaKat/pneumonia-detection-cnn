"""
Prueft die Nicht-Torch-Logik von rsna_train.py: Boxgeometrie, Eckenmaske,
geschichtete Metriken. Braucht keine GPU und kein Training.

Warum das hier wichtig ist: die Grad-CAM-Auswertung rechnet Bounding Boxes aus
dem 1024er-DICOM-Raster in das 224er-Modellraster um. Ein Faktorfehler faellt
in der fertigen Zahl NICHT auf -- eine Trefferquote von 0,4 sieht plausibel aus,
egal ob die Boxen an der richtigen Stelle liegen. Also wird die Geometrie hier
gegen von Hand gerechnete Faelle geprueft.

  python test_rsna_train.py
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

FAILED = []


def check(name: str, cond: bool, info: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FAILED.append(name)


def _stub_torch() -> None:
    """Attrappen NUR, wenn Torch wirklich fehlt (z.B. CI ohne GPU-Stack).

    Frueher wurde blind gestubbt, sobald `torch` nicht in `sys.modules` stand.
    Das sprengt jede Umgebung, in der Torch installiert IST: scipy fragt beim
    Import ueber array_api_compat `torch.Tensor` ab, findet am leeren Modul
    nichts und wirft einen AttributeError mitten im sklearn-Import. Also erst
    ehrlich probieren, dann ersetzen -- und der Ersatz bekommt `Tensor`.
    """
    try:
        import torch            # noqa: F401
        import torchvision      # noqa: F401
        return
    except ImportError:
        pass
    for name in ["torch", "torch.nn", "torch.utils", "torch.utils.data",
                 "torchvision", "torchvision.transforms", "torchvision.models"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["torch"].Tensor = type("Tensor", (), {})
    sys.modules["torch"].no_grad = lambda: (lambda f: f)
    sys.modules["torch"].device = lambda s: types.SimpleNamespace(type=s)
    sys.modules["torch.utils.data"].Dataset = object
    sys.modules["torch.utils.data"].DataLoader = object
    sys.modules["torchvision.models"].resnet18 = None
    sys.modules["torchvision.models"].ResNet18_Weights = None
    sys.modules["torchvision"].transforms = sys.modules["torchvision.transforms"]


_stub_torch()

from rsna_train import (BOX_SPACE, MaskCorners, load_boxes,  # noqa: E402
                        stratified_scores, youden_threshold)


def box_mask(boxes, size):
    """Dieselbe Geometrie wie in cam_vs_boxes -- hier isoliert nachgebaut."""
    s = size / BOX_SPACE
    mask = np.zeros((size, size), bool)
    for bx, by, bw, bh in boxes:
        x0, y0 = int(bx * s), int(by * s)
        x1, y1 = int((bx + bw) * s), int((by + bh) * s)
        mask[max(y0, 0):y1, max(x0, 0):x1] = True
    return mask


def test_box_geometry() -> None:
    print("\nBoxgeometrie 1024 -> 224")
    # Box ueber das gesamte Bild -> Maske komplett gesetzt
    m = box_mask([(0, 0, 1024, 1024)], 224)
    check("Vollbildbox deckt alles", m.all(), f"{m.mean():.3f}")

    # Rechtes unteres Viertel: x,y = 512, w,h = 512
    m = box_mask([(512, 512, 512, 512)], 224)
    check("Viertelbox deckt ~25 %", abs(m.mean() - 0.25) < 0.02, f"{m.mean():.3f}")
    check("Viertelbox sitzt unten rechts",
          m[200, 200] and not m[20, 20] and not m[20, 200] and not m[200, 20])

    # x ist die SPALTE, y die ZEILE -- klassischer Vertauschungsfehler
    m = box_mask([(0, 512, 1024, 512)], 224)     # ganze Breite, untere Haelfte
    check("x=Spalte, y=Zeile nicht vertauscht",
          m[200, 20] and m[200, 200] and not m[20, 20],
          "untere Haelfte gesetzt, obere nicht")

    # zwei getrennte Boxen (typisch: je Lungenfluegel eine)
    m = box_mask([(100, 300, 250, 400), (650, 300, 250, 400)], 224)
    check("zwei Boxen ergeben zwei Bereiche", 0.15 < m.mean() < 0.25, f"{m.mean():.3f}")

    # Box laeuft ueber den Rand hinaus -> darf nicht umbrechen
    m = box_mask([(900, 900, 400, 400)], 224)
    check("Box ueber den Rand bricht nicht um", m[-1, -1] and not m[0, 0])


def test_mask_corners() -> None:
    print("\nEckenmaske")
    from PIL import Image
    # Grundhelligkeit bewusst != 0, sonst ist "Median == 0" trivial erfuellt
    # und der Test wuerde eine Nullfuellung faelschlich durchwinken.
    a = np.full((100, 100), 80, np.uint8)
    a[40:60, 40:60] = 200            # heller Fleck in der Mitte
    a[:10, :10] = 255                # "Marker" in der Ecke
    out = np.asarray(MaskCorners(frac=0.18)(Image.fromarray(a)))
    check("Ecke ueberschrieben", out[:18, :18].max() < 255, f"max {out[:18, :18].max()}")
    check("Mitte unangetastet", out[50, 50] == 200)
    check("Fuellwert ist der Median, nicht 0", out[0, 0] == int(np.median(a)),
          f"{out[0, 0]} vs Median {int(np.median(a))}")
    check("Groesse unveraendert", out.shape == a.shape)


def test_stratified() -> None:
    print("\nGeschichtete Metriken")
    rng = np.random.default_rng(0)
    n = 600
    vp = np.array(["AP"] * (n // 2) + ["PA"] * (n // 2))
    y = rng.integers(0, 2, n).astype(float)
    # Score, der NUR die Projektion kennt -- gesamt informativ, je Schicht blind
    y[vp == "AP"] = (rng.random((vp == "AP").sum()) < 0.7).astype(float)
    y[vp == "PA"] = (rng.random((vp == "PA").sum()) < 0.15).astype(float)
    p_view = (vp == "AP").astype(float) + rng.normal(0, 1e-6, n)
    r = stratified_scores(y, p_view, vp, thr=0.5)
    check("Nur-Projektion-Score je Schicht ~0.5",
          abs(r["auc_AP"] - 0.5) < 0.05 and abs(r["auc_PA"] - 0.5) < 0.05,
          f"AP {r['auc_AP']:.3f} PA {r['auc_PA']:.3f}")
    check("geschichtete AUC liegt zwischen den Schichten",
          min(r["auc_AP"], r["auc_PA"]) <= r["auc_stratified"] <= max(r["auc_AP"], r["auc_PA"]))
    check("n je Schicht stimmt", r["n_AP"] == 300 and r["n_PA"] == 300)

    # echter Score -> beide Schichten deutlich ueber 0.5
    p_real = y * 0.6 + rng.random(n) * 0.4
    r2 = stratified_scores(y, p_real, vp, thr=0.5)
    check("informativer Score je Schicht > 0.8",
          r2["auc_AP"] > 0.8 and r2["auc_PA"] > 0.8,
          f"AP {r2['auc_AP']:.3f} PA {r2['auc_PA']:.3f}")


def test_threshold_per_view() -> None:
    """Schliesst die schichteigene Schwelle die Sensitivitaetsluecke?

    Nachgebaut wird die Lage aus dem ersten Lauf: gleiche Trennschaerfe in
    beiden Projektionen, aber sehr ungleiche Praevalenz. Genau dann laeuft eine
    feste Schwelle in AP und PA auf verschiedene Sensitivitaeten hinaus.
    """
    print("\nSchwelle je Projektion")
    rng = np.random.default_rng(0)

    def block(n, pos_rate, view):
        y = (rng.random(n) < pos_rate).astype(float)
        # identische Trennung in beiden Bloecken, nur verschoben:
        # der AP-Block liegt insgesamt hoeher, wie beim echten Modell
        shift = 0.25 if view == "AP" else 0.0
        p = np.clip(y * 0.45 + rng.normal(0.25, 0.12, n) + shift, 0, 1)
        return y, p, np.full(n, view)

    ya, pa, va = block(2000, 0.383, "AP")
    yp, pp, vp_ = block(2000, 0.093, "PA")
    y = np.r_[ya, yp]; p = np.r_[pa, pp]; vp = np.r_[va, vp_]

    thr_g = youden_threshold(y, p)
    thr_v = {v: youden_threshold(y[vp == v], p[vp == v]) for v in ("AP", "PA")}
    r = stratified_scores(y, p, vp, thr_g, thr_v)

    check("beide Schichten aehnlich trennscharf",
          abs(r["auc_AP"] - r["auc_PA"]) < 0.06,
          f"AP {r['auc_AP']:.3f} PA {r['auc_PA']:.3f}")
    check("globale Schwelle erzeugt eine Sens-Luecke", r["sens_gap"] > 0.15,
          f"{r['sens_gap']:.3f}")
    check("Schichtschwelle schliesst sie deutlich",
          r["sens_gap_strat"] < r["sens_gap"] / 2,
          f"{r['sens_gap']:.3f} -> {r['sens_gap_strat']:.3f}")
    check("beide Schwellen werden berichtet",
          "thr_AP" in r and "thr_PA" in r and r["thr_AP"] != r["thr_PA"])
    check("globale Werte bleiben erhalten",
          "sens_AP" in r and "sens_AP_strat" in r,
          "beide Varianten stehen nebeneinander")

    # Ohne Schichtschwellen darf nichts kaputtgehen
    r2 = stratified_scores(y, p, vp, thr_g)
    check("ohne thr_by_view weiterhin lauffaehig",
          "sens_gap" in r2 and "sens_gap_strat" not in r2)


def test_boxes_and_threshold() -> None:
    print("\nBox-CSV und Youden-Schwelle")
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        pd.DataFrame([
            {"patientId": "a", "x": np.nan, "y": np.nan, "width": np.nan,
             "height": np.nan, "Target": 0},
            {"patientId": "c", "x": 10, "y": 20, "width": 30, "height": 40, "Target": 1},
            {"patientId": "c", "x": 50, "y": 60, "width": 70, "height": 80, "Target": 1},
        ]).to_csv(d / "stage_2_train_labels.csv", index=False)
        b = load_boxes(d)
        check("nur Positive haben Boxen", set(b) == {"c"}, str(sorted(b)))
        check("beide Boxen erfasst", len(b["c"]) == 2)
        check("Reihenfolge x,y,w,h", b["c"][0] == (10.0, 20.0, 30.0, 40.0))

    y = np.array([0, 0, 0, 1, 1, 1.0])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    check("perfekt trennbar -> Schwelle im positiven Bereich",
          youden_threshold(y, p) == 0.7, str(youden_threshold(y, p)))


if __name__ == "__main__":
    test_box_geometry()
    test_mask_corners()
    test_stratified()
    test_threshold_per_view()
    test_boxes_and_threshold()
    print("\n" + ("ALLE TESTS BESTANDEN" if not FAILED
                  else f"{len(FAILED)} FEHLGESCHLAGEN: {FAILED}"))
    raise SystemExit(1 if FAILED else 0)
