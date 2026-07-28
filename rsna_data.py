"""
Zentrale Datendefinition fuer den RSNA Pneumonia Detection Challenge.

Alles, was "was ist eigentlich positiv?" beantwortet, steht HIER und nur hier.
Auf Kermany war das trivial (zwei Ordner), auf RSNA ist es eine Entscheidung
mit Folgen -- und verstreute Label-Logik ist der zuverlaessigste Weg, sich
selbst zu betruegen.

--------------------------------------------------------------------------
Die Labelentscheidung
--------------------------------------------------------------------------
RSNA liefert DREI Klassen (stage_2_detailed_class_info.csv):

  "Normal"                          ~8 850   unauffaelliger Thorax
  "No Lung Opacity / Not Normal"   ~11 800   auffaellig, aber KEINE Pneumonie
                                             (Erguss, Stauung, Narben, Geraete,
                                              Kardiomegalie, Fehlprojektion ...)
  "Lung Opacity"                    ~6 010   Infiltrat, mit Bounding Box

Wir trainieren binaer:  Lung Opacity = 1, alles andere = 0.

Warum die Mittelklasse zu den Negativen gehoert und nicht rausfliegt:

  * Klinisch ist die Frage "Pneumonie ja/nein", nicht "krank ja/nein".
    Wer die Mittelklasse entfernt, beantwortet die zweite Frage und behauptet,
    die erste beantwortet zu haben.
  * Genau das ist die Kermany-Lehre. Dort trennten sich die Klassen auch ueber
    Aufnahmesituation statt ueber Pathologie, und weil der Task zu leicht war,
    fiel es lange nicht auf. Die Mittelklasse zwingt das Modell, Infiltrat von
    anderer Pathologie zu unterscheiden -- nicht bloss auffaellig von leer.
  * Die Decke verschwindet damit. Literatur-AUC liegt fuer diese Definition bei
    ~0,85-0,90 statt bei 0,999. Erst da werden Unterschiede von 0,005 messbar.

Preis: die Aufgabe ist schwerer und die Zahlen sehen schlechter aus. Das ist
beabsichtigt. Wer die Kurven vergleichen will, kann `binary_label(..., mode=
"strict")` benutzen -- Normal vs. Lung Opacity, Mittelklasse verworfen. Das ist
eine ZUSATZauswertung ("wie viel des Signals steckt in krank-vs-gesund?"),
nicht der Haupttask.

--------------------------------------------------------------------------
Gruppierung
--------------------------------------------------------------------------
`patientId` ist in RSNA pro Bild eindeutig -- ein Patient, eine Aufnahme. Die
Kermany-Falle (mehrere Bilder desselben Kindes in Train UND Val) gibt es hier
nicht. Trotzdem wird ueberall nach `patientId` gruppiert: es kostet nichts und
haelt die Pipeline ehrlich, falls sich das je aendert.

ACHTUNG: stage_2_train_labels.csv hat MEHRERE Zeilen pro Bild, eine je
Bounding Box. Wer die Zeilen zaehlt statt der patientIds, bekommt ~30 200
statt 26 684 und eine verfaelschte Positivrate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- Konstanten

CLASS_NORMAL = "Normal"
CLASS_NOT_NORMAL = "No Lung Opacity / Not Normal"
CLASS_OPACITY = "Lung Opacity"
CLASSES3 = (CLASS_NORMAL, CLASS_NOT_NORMAL, CLASS_OPACITY)

#: DICOM-Felder, die als moegliche Confounder gepruefte werden. Alles davon
#: steht im Header, keins davon ist Radiologie.
HEADER_FIELDS = (
    "ViewPosition",        # AP vs. PA -- der Hauptverdaechtige, s.u.
    "PatientAge",
    "PatientSex",
    "Rows",
    "Columns",
    "PixelSpacing",        # wird zu pixel_spacing (float) aufgeloest
    "BodyPartExamined",
    "ConversionType",
    "PhotometricInterpretation",
    "Modality",
)

#: Der geerbte Shortcut aus NIH ChestX-ray14: AP-Aufnahmen entstehen ueberwiegend
#: am Bett mit mobilem Geraet, weil der Patient nicht mehr zum Stativ kann. Das
#: korreliert mit Schwere der Erkrankung, ohne irgendetwas ueber das Infiltrat zu
#: sagen. Vorteil gegenueber Kermany: es steht MESSBAR im Header und muss nicht
#: aus JPEG-Abmessungen rekonstruiert werden -- man kann also darauf matchen.
SUSPECTED_CONFOUNDERS = ("ViewPosition", "PatientAge", "PatientSex")


# ------------------------------------------------------------------- Labels

def binary_label(class3: pd.Series, mode: str = "clinical") -> pd.Series:
    """Dreiklassige RSNA-Klasse -> binaeres Label (mit NaN fuer 'verwerfen').

    mode="clinical" (Standard)  Lung Opacity = 1, Normal + Not-Normal = 0.
                                Der Haupttask. Kein Bild wird verworfen.
    mode="strict"               Lung Opacity = 1, Normal = 0,
                                Mittelklasse = NaN (verwerfen).
                                Nur fuer Zusatzauswertungen.
    """
    if mode == "clinical":
        return (class3 == CLASS_OPACITY).astype(float)
    if mode == "strict":
        out = pd.Series(float("nan"), index=class3.index, dtype=float)
        out[class3 == CLASS_OPACITY] = 1.0
        out[class3 == CLASS_NORMAL] = 0.0
        return out
    raise ValueError(f"unbekannter mode: {mode!r} (erlaubt: clinical, strict)")


def load_labels(csv_dir: Path, mode: str = "clinical") -> pd.DataFrame:
    """Liest die beiden Label-CSVs und faltet sie auf EINE Zeile je Bild.

    Rueckgabe: patientId, class3, target, label, n_boxes
      target   das offizielle 0/1 aus stage_2_train_labels.csv
      label    unser Label nach `mode` (bei mode="strict" sind Zeilen entfernt)
      n_boxes  Anzahl Bounding Boxes (0 fuer Negative) -- spaeter fuer die
               Grad-CAM-Auswertung, der eigentliche Grund fuer den Wechsel
    """
    csv_dir = Path(csv_dir)
    lab = pd.read_csv(csv_dir / "stage_2_train_labels.csv")
    cls = pd.read_csv(csv_dir / "stage_2_detailed_class_info.csv")

    # Eine Zeile je Box -> eine Zeile je Bild.
    boxes = (lab.groupby("patientId")
                .agg(target=("Target", "max"),
                     n_boxes=("x", lambda s: int(s.notna().sum())))
                .reset_index())
    cls = cls.drop_duplicates("patientId")

    df = boxes.merge(cls[["patientId", "class"]], on="patientId", how="left")
    df = df.rename(columns={"class": "class3"})

    unknown = set(df["class3"].dropna().unique()) - set(CLASSES3)
    if unknown:
        raise ValueError(f"unerwartete Klassennamen in der CSV: {unknown}")

    # Konsistenzprobe: Target==1 muss genau Lung Opacity sein. Faellt das um,
    # stimmt die Annahme ueber den Datensatz nicht mehr.
    mism = int((df["target"].eq(1) != df["class3"].eq(CLASS_OPACITY)).sum())
    if mism:
        raise ValueError(f"{mism} Zeilen: Target passt nicht zu 'Lung Opacity'")

    df["label"] = binary_label(df["class3"], mode=mode)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["group"] = df["patientId"]      # ein Bild je Patient, s. Modul-Docstring
    return df.reset_index(drop=True)


# ------------------------------------------------------------ DICOM-Header

def _age_to_years(raw) -> float:
    """PatientAge kommt als '058Y', '058', gelegentlich Monate ('006M')."""
    if raw is None:
        return float("nan")
    s = str(raw).strip().upper()
    digits = "".join(c for c in s if c.isdigit())
    if not digits:
        return float("nan")
    v = float(digits)
    if s.endswith("M"):
        v /= 12.0
    elif s.endswith("W"):
        v /= 52.0
    elif s.endswith("D"):
        v /= 365.0
    # RSNA enthaelt vereinzelt Unsinn wie 148 oder 413 Jahre.
    return v if 0 < v < 120 else float("nan")


def _first_spacing(raw) -> float:
    """Erster Wert von PixelSpacing, egal in welcher Gestalt er ankommt.

    pydicom liefert `MultiValue` -- das ist KEINE list und keine tuple, sondern
    eine eigene Sequence-Klasse. Ein `isinstance(v, (list, tuple))` schlaegt
    deshalb still fehl und die Spalte wird komplett NaN (so passiert im ersten
    Probelauf). Nach dem Umweg ueber den CSV-Cache kommt derselbe Wert dagegen
    als String '[0.139, 0.139]' zurueck. Beides muss hier durchgehen.
    """
    if raw is None:
        return float("nan")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        parts = raw.strip().strip("[]()").split(",")
        try:
            return float(parts[0].strip().strip("'\""))
        except (ValueError, IndexError):
            return float("nan")
    try:                       # MultiValue, list, tuple, numpy array ...
        return float(raw[0])
    except (TypeError, ValueError, IndexError, KeyError):
        return float("nan")


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    """Rohe Header-Spalten -> auswertbare Spalten. Idempotent."""
    df = df.copy()
    df["age_years"] = df["PatientAge"].map(_age_to_years)
    df["pixel_spacing"] = df["PixelSpacing"].map(_first_spacing)
    for c in ("Rows", "Columns"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("ViewPosition", "PatientSex"):
        df[c] = df[c].astype("string").fillna("UNKNOWN")
    return df


def scan_headers(dicom_dir: Path, cache: Path | None = None,
                 limit: int | None = None) -> pd.DataFrame:
    """Liest NUR die Header aller DICOMs (kein Pixeldaten-Dekodieren).

    26 684 Dateien, `stop_before_pixels=True` -> wenige Minuten statt Stunden.
    Ergebnis wird in `cache` als CSV abgelegt; ein zweiter Aufruf liest die CSV.
    """
    if cache is not None and Path(cache).exists():
        # Abgeleitete Spalten werden neu berechnet, nicht aus der CSV uebernommen:
        # sonst ueberlebt ein Parser-Fehler den Fix im Cache.
        return _derive(pd.read_csv(cache))

    import pydicom  # nur hier importiert, damit der Rest ohne pydicom testbar ist

    dicom_dir = Path(dicom_dir)
    files = sorted(dicom_dir.glob("*.dcm"))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"Keine .dcm-Dateien unter {dicom_dir}")

    rows = []
    for i, f in enumerate(files, 1):
        ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        row = {"patientId": f.stem}
        for k in HEADER_FIELDS:
            row[k] = getattr(ds, k, None)
        rows.append(row)
        if i % 2000 == 0:
            print(f"  {i}/{len(files)} Header gelesen")

    df = _derive(pd.DataFrame(rows))

    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        print(f"Header-Cache geschrieben: {cache}")
    return df
