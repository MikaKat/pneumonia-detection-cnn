"""
Central data definition for the RSNA Pneumonia Detection Challenge.

Everything that answers "what actually counts as positive?" lives HERE and
nowhere else. On Kermany that was trivial (two folders); on RSNA it is a
decision with consequences, and label logic scattered across several files is
the most reliable way to deceive oneself.

--------------------------------------------------------------------------
The labelling decision
--------------------------------------------------------------------------
RSNA ships THREE classes (stage_2_detailed_class_info.csv):

  "Normal"                          ~8 850   unremarkable chest
  "No Lung Opacity / Not Normal"   ~11 800   abnormal, but NO pneumonia
                                             (effusion, congestion, scarring,
                                              devices, cardiomegaly, bad
                                              projection ...)
  "Lung Opacity"                    ~6 010   infiltrate, with bounding box

Training is binary:  Lung Opacity = 1, everything else = 0.

Why the middle class belongs with the negatives instead of being dropped:

  * Clinically the question is "pneumonia yes/no", not "ill yes/no". Removing
    the middle class answers the second question while claiming to have
    answered the first.
  * That is exactly the Kermany lesson. There, too, the classes separated by
    imaging situation rather than by pathology, and because the task was too
    easy it went unnoticed for a long time. The middle class forces the model
    to distinguish an infiltrate from other pathology, not merely abnormal
    from empty.
  * The ceiling disappears with it. For this definition the literature AUC is
    around 0.85-0.90 instead of 0.999. AUC is the probability that a random
    pneumonia case is ranked above a random non-case, the same quantity as a
    c-statistic. Away from the ceiling, differences of 0.005 become measurable.

The price: the task is harder and the numbers look worse. That is intended.
For comparison with the older curves, `binary_label(..., mode="strict")` gives
Normal vs. Lung Opacity with the middle class discarded. That is an ADDITIONAL
evaluation ("how much of the signal sits in ill-vs-healthy?"), not the main
task.

--------------------------------------------------------------------------
Grouping
--------------------------------------------------------------------------
In RSNA, `patientId` is unique per image: one patient, one acquisition. The
Kermany trap (several images of the same child in train AND val) does not exist
here. Even so, everything groups by `patientId`. It costs nothing and keeps the
pipeline honest should that ever change.

NOTE: stage_2_train_labels.csv has SEVERAL rows per image, one per bounding
box. Counting rows instead of patientIds yields ~30 200 instead of 26 684 and a
distorted positive rate.

--------------------------------------------------------------------------
Interpreting the output
--------------------------------------------------------------------------
`load_labels` returns one row per image, so `len(df)` is the image count, not
the box count, and `df["label"].mean()` is the positive rate of the task as
defined by `mode`. With mode="clinical" every image is kept and the row count
matches the number of DICOMs. With mode="strict" the middle class is gone, and
both the row count and the positive rate change; the two modes are not
comparable to each other. `n_boxes` is 0 for every negative by construction, so
a positive without a box would mean the box table and the class table disagree.
Two checks are hard failures rather than warnings: an unknown class name in the
CSV, and any row where `Target` and "Lung Opacity" disagree. Either one
falsifies the assumption this module rests on, and the labelling has to be
re-examined before any training result means anything. `scan_headers` returns
one row per DICOM with the header fields plus the derived columns `age_years`
and `pixel_spacing`; a column that is entirely NaN indicates a parsing problem,
not an empty header.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------- Constants

CLASS_NORMAL = "Normal"
CLASS_NOT_NORMAL = "No Lung Opacity / Not Normal"
CLASS_OPACITY = "Lung Opacity"
CLASSES3 = (CLASS_NORMAL, CLASS_NOT_NORMAL, CLASS_OPACITY)

#: DICOM fields that are checked as possible confounders. All of them are in
#: the header, none of them is radiology.
HEADER_FIELDS = (
    "ViewPosition",        # AP vs. PA, the prime suspect (see below)
    "PatientAge",
    "PatientSex",
    "Rows",
    "Columns",
    "PixelSpacing",        # is resolved into pixel_spacing (float)
    "BodyPartExamined",
    "ConversionType",
    "PhotometricInterpretation",
    "Modality",
)

#: The shortcut inherited from NIH ChestX-ray14: AP images are predominantly
#: taken at the bedside with a mobile unit, because the patient can no longer
#: reach the stand. That correlates with severity of illness without saying
#: anything about the infiltrate. Advantage over Kermany: here it sits
#: MEASURABLY in the header and need not be reconstructed from JPEG dimensions,
#: so it can be matched on.
SUSPECTED_CONFOUNDERS = ("ViewPosition", "PatientAge", "PatientSex")


# ------------------------------------------------------------------- Labels

def binary_label(class3: pd.Series, mode: str = "clinical") -> pd.Series:
    """Three-class RSNA class -> binary label (with NaN for 'discard').

    mode="clinical" (default)   Lung Opacity = 1, Normal + Not-Normal = 0.
                                The main task. No image is discarded.
    mode="strict"               Lung Opacity = 1, Normal = 0,
                                middle class = NaN (discard).
                                Only for additional evaluations.
    """
    if mode == "clinical":
        return (class3 == CLASS_OPACITY).astype(float)
    if mode == "strict":
        out = pd.Series(float("nan"), index=class3.index, dtype=float)
        out[class3 == CLASS_OPACITY] = 1.0
        out[class3 == CLASS_NORMAL] = 0.0
        return out
    raise ValueError(f"unknown mode: {mode!r} (allowed: clinical, strict)")


def load_labels(csv_dir: Path, mode: str = "clinical") -> pd.DataFrame:
    """Reads the two label CSVs and folds them onto ONE row per image.

    Returns: patientId, class3, target, label, n_boxes
      target   the official 0/1 from stage_2_train_labels.csv
      label    the label as defined by `mode` (with mode="strict", rows are
               removed)
      n_boxes  number of bounding boxes (0 for negatives), used later for the
               Grad-CAM evaluation (which region of the image the model
               responded to), the actual reason for the switch
    """
    csv_dir = Path(csv_dir)
    lab = pd.read_csv(csv_dir / "stage_2_train_labels.csv")
    cls = pd.read_csv(csv_dir / "stage_2_detailed_class_info.csv")

    # One row per box -> one row per image.
    boxes = (lab.groupby("patientId")
                .agg(target=("Target", "max"),
                     n_boxes=("x", lambda s: int(s.notna().sum())))
                .reset_index())
    cls = cls.drop_duplicates("patientId")

    df = boxes.merge(cls[["patientId", "class"]], on="patientId", how="left")
    df = df.rename(columns={"class": "class3"})

    unknown = set(df["class3"].dropna().unique()) - set(CLASSES3)
    if unknown:
        raise ValueError(f"unexpected class names in the CSV: {unknown}")

    # Consistency check: Target==1 must be exactly Lung Opacity. If this fails,
    # the assumption about the dataset no longer holds.
    mism = int((df["target"].eq(1) != df["class3"].eq(CLASS_OPACITY)).sum())
    if mism:
        raise ValueError(f"{mism} rows: Target does not match 'Lung Opacity'")

    df["label"] = binary_label(df["class3"], mode=mode)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["group"] = df["patientId"]      # one image per patient, see module docstring
    return df.reset_index(drop=True)


# ------------------------------------------------------------ DICOM headers

def _age_to_years(raw) -> float:
    """PatientAge arrives as '058Y', '058', occasionally months ('006M')."""
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
    # RSNA contains occasional nonsense such as 148 or 413 years.
    return v if 0 < v < 120 else float("nan")


def _first_spacing(raw) -> float:
    """First value of PixelSpacing, in whatever shape it arrives.

    pydicom returns a `MultiValue`, which is NOT a list and not a tuple but a
    Sequence class of its own. An `isinstance(v, (list, tuple))` therefore
    fails silently and the column becomes entirely NaN. After the detour
    through the CSV cache, by contrast, the same value comes back as the string
    '[0.139, 0.139]'. Both have to pass through here.
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
    """Raw header columns -> analysable columns. Idempotent."""
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
    """Reads ONLY the headers of all DICOMs (no pixel data decoding).

    26 684 files, `stop_before_pixels=True` -> a few minutes instead of hours.
    The result is stored in `cache` as CSV; a second call reads that CSV.
    """
    if cache is not None and Path(cache).exists():
        # Derived columns are recomputed rather than taken from the CSV:
        # otherwise a parser bug would survive its own fix inside the cache.
        return _derive(pd.read_csv(cache))

    import pydicom  # imported here only, so the rest is testable without pydicom

    dicom_dir = Path(dicom_dir)
    files = sorted(dicom_dir.glob("*.dcm"))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"No .dcm files under {dicom_dir}")

    rows = []
    for i, f in enumerate(files, 1):
        ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        row = {"patientId": f.stem}
        for k in HEADER_FIELDS:
            row[k] = getattr(ds, k, None)
        rows.append(row)
        if i % 2000 == 0:
            print(f"  {i}/{len(files)} headers read")

    df = _derive(pd.DataFrame(rows))

    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        print(f"header cache written: {cache}")
    return df
