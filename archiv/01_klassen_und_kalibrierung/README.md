# Where the errors sit, and whether the score is a probability

Two questions, no training, about six seconds of compute on predictions that
were already on disk. Both answers were written down before the run and both
came true.

## Where the errors sit

RSNA labels each film `Normal`, `Lung Opacity` (with a box) or `No Lung Opacity
/ Not Normal`, meaning abnormal but not pneumonia. That middle class is where
the false positives should be if the model has learned anything clinical.

Error rate at the Youden point of each fold's selection split:

| Class | n | pooled | AP only | PA only |
| --- | --- | --- | --- | --- |
| Normal | 1504 | 0.019 ± 0.003 | 0.062 | 0.009 |
| Abnormal, no infiltrate | 2039 | 0.373 ± 0.028 | 0.533 | 0.215 |
| Infiltrate | 1030 | 0.813 ± 0.034 | 0.882 | 0.574 |
| ratio middle to normal | | 20.0x | 8.6x | 24.0x |

The finding survives the breakdown, which was the actual test: clean films are
81.6 percent PA, so a pooled number would have flattered it. Within AP films the
false alarm rate on clean chests is 6.2 percent, not 1.9.

The measure of choice is the ratio of error rates, not the share of false
positives. 96.5 percent of the false positives are in the middle class, but that
class is already 57.5 percent of all negatives.

What could not be answered: whether the heat map on those errors points at the
other pathology that is actually there. RSNA draws boxes for infiltrates only.

## Whether the score is a probability

It is not. Raw, over all folds:

| | observed | predicted |
| --- | --- | --- |
| AP | 0.383 | 0.564 |
| PA | 0.093 | 0.186 |
| total | 0.225 | 0.358 |

In readable units the typical distance to the diagonal is 0.182 probability
points. The clearest single example: on AP films the raw model says 0.80 where
0.49 are actually positive. A two-parameter Platt curve, fitted on the inner
selection split, brings the typical distance to 0.017 and the largest single
deviation from 0.382 to 0.079.

The cause was tested rather than asserted. A class weight can only shift the
logit, and a pure shift removes 94 percent of the squared miscalibration, so
most of the height is `pos_weight`. But the free slope is 0.756 ± 0.050, clearly
below 1, and the required shift varies from 2.74 to 4.91 across folds while the
class ratio is 3.44 in every one of them. A constant cause does not produce a
varying effect: `pos_weight` explains the height, not the shape and not the
spread.

Fitting Platt per projection instead of jointly is marginally worse (0.000305
against 0.000287 reliability). That is a piece of luck rather than a design
success, and the shipped model depends on it: the web app receives a PNG with no
DICOM header and could not pick a per-projection curve even if one were better.

## The operating points, and why the gap widens

| Operating point | | sensitivity | specificity |
| --- | --- | --- | --- |
| Youden | pooled | 0.813 ± 0.034 | 0.778 ± 0.017 |
| | AP | 0.882 | 0.568 |
| | PA | 0.574 | 0.897 |
| specificity 0.95 | AP | 0.563 | 0.881 |
| | PA | 0.200 | 0.985 |

The gap between the projections grows at stricter operating points rather than
shrinking. At specificity 0.95 the model misses four fifths of the infiltrates
on standing PA films.

## Contents and re-running

`predictions_klassen/` holds the per-class breakdown, the calibration curves and
`kalibrierung.png`.

```powershell
venv\Scripts\python.exe rsna\befunde\rsna_klassen_kalibrierung.py --pred-dir archiv\00_erste_laeufe_und_nebenanalysen\predictions_rsna_base --out-dir archiv\01_klassen_und_kalibrierung\predictions_klassen
venv\Scripts\python.exe rsna\befunde\rsna_phase3_pruefung.py --pred-dir archiv\00_erste_laeufe_und_nebenanalysen\predictions_rsna_base
```

The second script is the check that recomputes every number of the first from
the raw predictions and stops on disagreement. It found no arithmetic error and
one reporting error: every headline had been pooled.
