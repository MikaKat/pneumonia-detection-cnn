"""
Prueft die RSNA-Label- und Header-Logik OHNE Datensatz und OHNE pydicom.

Zweck: der erste echte Lauf liest 26 684 DICOM-Header. Wenn er nach zwanzig
Minuten an einem Tippfehler in der Label-Faltung stirbt, ist das aergerlich;
wenn er stattdessen still das falsche Label baut, ist es schlimmer. Deshalb
laeuft alles Nicht-IO hier gegen synthetische Tabellen.

  python test_rsna_data.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_data import (CLASS_NORMAL, CLASS_NOT_NORMAL, CLASS_OPACITY,
                       _age_to_years, binary_label, load_labels)
from rsna_metadata_leak_check import categorical_table, single_auc

FAILED = []


def check(name: str, cond: bool, info: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {info}" if info else ""))
    if not cond:
        FAILED.append(name)


def fake_csvs(d: Path) -> None:
    """Drei Bilder: eins normal, eins Mittelklasse, eins mit ZWEI Boxen."""
    pd.DataFrame([
        {"patientId": "a", "x": np.nan, "y": np.nan, "width": np.nan,
         "height": np.nan, "Target": 0},
        {"patientId": "b", "x": np.nan, "y": np.nan, "width": np.nan,
         "height": np.nan, "Target": 0},
        {"patientId": "c", "x": 10, "y": 20, "width": 30, "height": 40, "Target": 1},
        {"patientId": "c", "x": 50, "y": 60, "width": 70, "height": 80, "Target": 1},
    ]).to_csv(d / "stage_2_train_labels.csv", index=False)

    pd.DataFrame([
        {"patientId": "a", "class": CLASS_NORMAL},
        {"patientId": "b", "class": CLASS_NOT_NORMAL},
        {"patientId": "c", "class": CLASS_OPACITY},
        {"patientId": "c", "class": CLASS_OPACITY},   # doppelt, wie im Original
    ]).to_csv(d / "stage_2_detailed_class_info.csv", index=False)


def test_labels() -> None:
    print("\nLabel-Faltung und Modi")
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        fake_csvs(d)

        cl = load_labels(d, mode="clinical")
        check("eine Zeile je Bild (nicht je Box)", len(cl) == 3, f"n={len(cl)}")
        check("Boxen gezaehlt",
              cl.set_index("patientId")["n_boxes"].to_dict() == {"a": 0, "b": 0, "c": 2})
        lut = cl.set_index("patientId")["label"].to_dict()
        check("Mittelklasse zaehlt als negativ", lut == {"a": 0, "b": 0, "c": 1}, str(lut))
        check("group == patientId", (cl["group"] == cl["patientId"]).all())

        st = load_labels(d, mode="strict")
        check("strict verwirft die Mittelklasse",
              set(st["patientId"]) == {"a", "c"}, str(sorted(st['patientId'])))

        # Inkonsistente CSV muss krachen, nicht still durchlaufen.
        bad = pd.read_csv(d / "stage_2_train_labels.csv")
        bad.loc[bad["patientId"] == "a", "Target"] = 1
        bad.to_csv(d / "stage_2_train_labels.csv", index=False)
        try:
            load_labels(d)
            check("inkonsistentes Target wird erkannt", False)
        except ValueError:
            check("inkonsistentes Target wird erkannt", True)


def test_binary_label() -> None:
    print("\nbinary_label direkt")
    s = pd.Series([CLASS_NORMAL, CLASS_NOT_NORMAL, CLASS_OPACITY])
    check("clinical -> 0,0,1", list(binary_label(s, "clinical")) == [0.0, 0.0, 1.0])
    strict = binary_label(s, "strict")
    check("strict -> 0,NaN,1",
          strict[0] == 0.0 and np.isnan(strict[1]) and strict[2] == 1.0)
    try:
        binary_label(s, "irgendwas")
        check("unbekannter Modus kracht", False)
    except ValueError:
        check("unbekannter Modus kracht", True)


def test_age() -> None:
    print("\nPatientAge-Parsing")
    cases = {"058Y": 58.0, "058": 58.0, "  61Y ": 61.0, "006M": 0.5,
             "000Y": float("nan"), "148Y": float("nan"), "": float("nan"),
             None: float("nan"), "abc": float("nan")}
    for raw, want in cases.items():
        got = _age_to_years(raw)
        ok = (np.isnan(got) and np.isnan(want)) or abs(got - want) < 1e-9
        check(f"{raw!r} -> {want}", ok, f"got {got}")


def test_auc_helpers() -> None:
    print("\nAUC-Hilfsfunktionen")
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(200)]
    s = np.r_[rng.normal(0, 1, 200), rng.normal(1.5, 1, 200)]
    a = single_auc(y, s)
    check("informatives Merkmal deutlich ueber 0.5", a > 0.8, f"AUC {a:.3f}")
    check("Richtung egal", abs(single_auc(y, -s) - a) < 1e-12)
    check("konstantes Merkmal -> NaN", np.isnan(single_auc(y, np.ones(400))))
    s2 = s.copy(); s2[:100] = np.nan
    check("NaNs werden ausgelassen, nicht als 0 gewertet",
          not np.isnan(single_auc(y, s2)))
    check("reines Rauschen ~0.5",
          abs(single_auc(y, rng.normal(0, 1, 400)) - 0.5) < 0.12)

    print("\ncategorical_table")
    df = pd.DataFrame({"label": [1, 1, 0, 0, 0], "ViewPosition": list("AAAPP")})
    t = categorical_table(df, "ViewPosition")
    check("Positivrate je Auspraegung",
          abs(t.loc["A", "pos_rate"] - 2 / 3) < 1e-9 and t.loc["P", "pos_rate"] == 0.0)
    check("nach Haeufigkeit sortiert", list(t.index) == ["A", "P"])


if __name__ == "__main__":
    test_labels()
    test_binary_label()
    test_age()
    test_auc_helpers()
    print("\n" + ("ALLE TESTS BESTANDEN" if not FAILED
                  else f"{len(FAILED)} FEHLGESCHLAGEN: {FAILED}"))
    raise SystemExit(1 if FAILED else 0)
