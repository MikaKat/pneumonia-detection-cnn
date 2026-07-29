"""
Confounder check for RSNA. RUNS BEFORE THE FIRST TRAINING RUN.

This is the one lesson carried over from the Kermany phase: there, training came
first, then Grad-CAM, then cropping, and only after weeks did it emerge that the
JPEG image dimensions alone separate the classes at AUC 0.915. Every model number
produced before that had to be read with a caveat. So this time the order is
reversed.

The question is: how far does a classifier get that sees NOT A SINGLE PIXEL, only
the DICOM header? The measure is AUC, the probability that a random pneumonia case
is ranked above a random non-case (the same quantity as a c-statistic). That number
is the lower bound the image model has to beat in order to have learned any
radiology at all.

Expectations stated up front (so that the result is not explained to fit after
the fact):

  * Rows/Columns:  constant 1024x1024. AUC ~0.50. The Kermany confounder does
    not exist here, one of the reasons for the switch. If something does turn
    up, the assumption is wrong and the plan has to be redone.
  * ViewPosition:  AP vs. PA. AP predominantly means a bedside image taken with
    a mobile unit, hence sicker patients. Expectation clearly > 0.50.
    This is a REAL confounder inherited from NIH ChestX-ray14.
  * PatientAge:    pneumonia clusters at the extremes of age. Expectation
    ~0.55-0.65.
  * combined:      probably 0.60-0.70. Clearly less than Kermany's 0.915, but
    far above chance, and for that reason later analyses have to match or
    stratify on age and ViewPosition, not on a proxy.

A high value does NOT mean the dataset is unusable. Age and acquisition type do
correlate with pneumonia in reality; a radiologist knows this too. It means that
a raw AUC contains that share, and that the reported number has to be matched.

Interpreting the output
-----------------------
The headline figure is the AUC of the header-only classifier under grouped
cross-validation. The null value is 0.5: at 0.5 the DICOM header betrays nothing
about the label, and a model's AUC can be read at face value. Anything above 0.5
is the score a pixel-blind classifier already achieves, and every later model
result in the project is reported against it. A model AUC that fails to clear it
has demonstrated nothing about lungs. On this dataset the header alone reaches
0.729, almost all of it projection.

The per-feature table localises the effect. A feature whose two medians coincide
is constant and cannot be a confounder; a feature with an AUC well above 0.5 is a
candidate shortcut that later evaluations have to stratify on. The categorical
tables show the positive rate per level, and the ViewPosition x class
cross-tabulation separates the two possible explanations: an effect tied to
pneumonia specifically, or an effect tied to being ill at all.

CLI:
  python rsna_metadata_leak_check.py \
      --dicom data/rsna/stage_2_train_images \
      --csv   data/rsna \
      --out   qc/rsna
"""

from __future__ import annotations

import _repo_path  # noqa: F401  (setzt sys.path fuer die Nachbarordner)

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from rsna_data import CLASSES3, load_labels, scan_headers

NUMERIC = ["age_years", "Rows", "Columns", "pixel_spacing"]
CATEGORICAL = ["ViewPosition", "PatientSex"]


def single_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Direction-independent AUC of one feature; NaNs are left out.

    Mirroring the value at 1 - a makes it a pure measure of separability: a
    feature that runs lower in the positive cases carries exactly as much
    information as one that runs higher, and only the amount matters here.
    Anything above 0.5 is information a model could exploit without ever
    looking at the lung.
    """
    ok = ~np.isnan(s)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2 or len(np.unique(s[ok])) < 2:
        return float("nan")
    a = roc_auc_score(y[ok], s[ok])
    return max(a, 1 - a)


def categorical_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Positive rate per level, more honest than an in-sample target encoding.

    A target encoding fitted on the same rows would hand each level its own label
    mean back as a score and thereby overstate the leak. The plain table keeps
    size and positive rate visible separately, so a level that looks extreme
    because it holds twelve images cannot be mistaken for a confounder.
    """
    t = df.groupby(col, dropna=False).agg(n=("label", "size"),
                                          pos_rate=("label", "mean"))
    return t.sort_values("n", ascending=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dicom", type=Path, default=Path("data/rsna/stage_2_train_images"))
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--out", type=Path, default=Path("qc/rsna"))
    p.add_argument("--mode", default="clinical", choices=["clinical", "strict"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N DICOMs. Trial run, NOT representative: "
                        "the file order is correlated with the label (see below)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lab = load_labels(args.csv, mode=args.mode)
    hdr = scan_headers(args.dicom, cache=args.out / "dicom_headers.csv",
                       limit=args.limit)
    df = lab.merge(hdr, on="patientId", how="inner")
    if df.empty:
        raise SystemExit("No image present in both sources. Are the paths right?")

    n_only_csv = len(lab) - len(df)
    y = df["label"].values
    if args.limit:
        print("\n  CAUTION: --limit takes the first N files in sort order,")
        print("  and that order is NOT random: the positive rate varies across")
        print("  the sorted file list between 0.136 and 0.373 (0.225 overall).")
        print("  The patientId UUIDs therefore carry batch structure. Numbers")
        print("  from a --limit run show that the code runs, nothing more.")
    print(f"\n{len(df)} images with label and header"
          f"{f' ({n_only_csv} only in the CSV)' if n_only_csv else ''}")
    print(f"Mode '{args.mode}': positive {int(y.sum())} ({y.mean():.3f}), "
          f"negative {int((y == 0).sum())}")
    print("\nThree-class distribution:")
    for c in CLASSES3:
        n = int((df["class3"] == c).sum())
        if n:
            print(f"  {c:<32}{n:>7}  ({n / len(df):.3f})")

    # ---- numeric single features ----
    print(f"\n{'Merkmal':<16}{'neg (Median)':>14}{'pos (Median)':>14}{'AUC':>8}")
    print("-" * 52)
    aucs = {}
    for f in NUMERIC:
        s = pd.to_numeric(df[f], errors="coerce").values.astype(float)
        aucs[f] = single_auc(y, s)
        med = [np.nan if np.isnan(s[y == v]).all() else np.nanmedian(s[y == v])
               for v in (0, 1)]      # nanmedian warns on entirely empty columns
        note = ""
        if np.isnan(aucs[f]) and not np.isnan(med[0]) and med[0] == med[1]:
            note = "  <- constant, no confounder possible"
        print(f"{f:<16}{med[0]:>14.3f}{med[1]:>14.3f}{aucs[f]:>8.3f}{note}")

    # ---- categorical features ----
    for c in CATEGORICAL:
        print(f"\n{c}: positive rate per level")
        t = categorical_table(df, c)
        for lvl, row in t.iterrows():
            print(f"  {str(lvl):<12}{int(row['n']):>8}{row['pos_rate']:>10.3f}")
        if len(t) == 2:  # binary -> directly readable as an AUC
            ind = (df[c] == t.index[0]).astype(float).values
            aucs[c] = single_auc(y, ind)
            print(f"  -> as a single feature: AUC {aucs[c]:.3f}")

    # Cross-tabulation of projection x three-class label: shows whether the
    # ViewPosition effect is attached to pneumonia or to "being ill at all".
    if df["ViewPosition"].nunique() > 1:
        print("\nViewPosition x class (row proportions):")
        ct = pd.crosstab(df["ViewPosition"], df["class3"], normalize="index")
        print(ct.round(3).to_string())

    # ---- combined, patient-grouped CV ----
    X = df[NUMERIC].apply(pd.to_numeric, errors="coerce")
    for c in CATEGORICAL:
        d = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True)
        X = pd.concat([X, d], axis=1)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0).values.astype(float)

    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=0)
    prob = cross_val_predict(GradientBoostingClassifier(random_state=0), X, y,
                             groups=df["group"].values, cv=cv,
                             method="predict_proba")[:, 1]
    auc = roc_auc_score(y, prob)
    df["header_score"] = prob

    print(f"\n>> Header-only classifier, {args.folds}-fold grouped CV: "
          f"AUC = {auc:.3f}")
    print("   No pixel was seen. The image model has to beat this number,")
    print("   and the reported model AUC has to be matched on these variables.")
    print(f"   (Kermany for comparison: 0.915 from the JPEG dimensions alone.)")

    # ---- plot ----
    plot_cols = [f for f in NUMERIC if not np.isnan(aucs.get(f, np.nan))]
    fig, axes = plt.subplots(1, len(plot_cols) + 1,
                             figsize=(4 * (len(plot_cols) + 1), 3.6))
    for ax, f in zip(axes, plot_cols):
        s = pd.to_numeric(df[f], errors="coerce")
        for lab_v, name in [(0, "no infiltrate"), (1, "Lung Opacity")]:
            ax.hist(s[y == lab_v].dropna(), bins=40, alpha=0.55,
                    label=name, density=True)
        ax.set_title(f"{f}  (AUC {aucs[f]:.3f})")
        ax.legend(fontsize=7)
    for lab_v, name in [(0, "no infiltrate"), (1, "Lung Opacity")]:
        axes[-1].hist(prob[y == lab_v], bins=40, alpha=0.55, label=name, density=True)
    axes[-1].set_title(f"header score (AUC {auc:.3f})")
    axes[-1].legend(fontsize=7)
    fig.suptitle(f"RSNA metadata leak, mode '{args.mode}': "
                 f"header alone separates at AUC {auc:.3f}")
    fig.tight_layout()
    fig.savefig(args.out / "rsna_metadata_leak.png", dpi=130)

    df.to_csv(args.out / "rsna_metadata_features.csv", index=False)
    print(f"\nsaved: {args.out}/rsna_metadata_features.csv, "
          f"rsna_metadata_leak.png, dicom_headers.csv")


if __name__ == "__main__":
    main()
