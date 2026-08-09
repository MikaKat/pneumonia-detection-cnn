# Does the heat map point at the pathology?

This is where the project changed its mind, and the correction is worth more
than the original claim.

## The claim that had to be withdrawn

The first version reported that the Grad-CAM peak lands inside an annotated box
in 0.539 of cases against a chance value of 0.117, a factor of 4.6. The
arithmetic was right and the denominator was wrong.

Box area is the chance value of a pointer that lands anywhere with equal
probability, including the shoulders, the image border and the air beside the
patient. Nobody guesses like that. Opacities sit in the lung fields, and the
lung fields sit in the same place in every chest radiograph.

## The opponent that had to be beaten instead

Every box of a fold's training part drawn into one grid and averaged gives the
location prior: one fixed map, identical for every image, that knows nothing
except where opacities usually sit. Beside it the lung map, the U-Net
segmentation of the individual image, because "point at the lung" is the trivial
solution.

The measure is the area under the curve over pixels inside the lung mask: draw
one pixel inside a box and one outside, how often is the map higher inside. Its
chance value is exactly 0.5 whatever the boxes cover, which is what makes it
comparable across crops and resolutions.

Five folds, all 5,154 positive validation images, no retraining:

| | point AUC in the lung | hit rate |
| --- | --- | --- |
| Location prior | 0.752 | 0.571 |
| Model | 0.704 | 0.542 |
| Lung map | 0.701 | 0.533 |
| Chance | 0.500 | 0.117 |

Paired over folds the model sits 0.048 ± 0.013 below the prior, t = -8.11. A
fixed anatomical map points at the pathology more reliably than the trained
model does.

## The part that matters more than the average

The prior can only encode the average position. If the model had learned nothing
but anatomy, the two would rise and fall on the same images. Binned by how
typical the box position is, using the prior's own score as the axis:

| | location prior | lung map | model |
| --- | --- | --- | --- |
| most atypical fifth | 0.512 | 0.667 | 0.720 |
| middle fifth | 0.782 | 0.711 | 0.707 |
| most typical fifth | 0.921 | 0.712 | 0.685 |

The prior swings by construction. The model does not move. Its per-image score
correlates with the prior's at r = -0.05 and with the lung map's at -0.02, while
three trained variants correlate with each other at 0.49 to 0.67. Two separate
families of information, and the model belongs to neither anatomical one.

On the 571 images where the prior falls below chance at 0.442, the model reaches
0.711 and beats the segmentation of the same patient's lungs by 0.061 ± 0.009,
t = +15.8, in the same direction in all five folds. Mask quality is flat across
the bins (box inside lung 0.859 to 0.879), so this is not a segmentation
artefact.

Both short versions are wrong. "The heat map localises 4.6 times better than
chance" answers the wrong question, and "the heat map does not localise" is
refuted by the control against the lung mask.

## A null result worth keeping

RSNA films carry burnt-in markers, and `PORTABLE` is printed on the AP films.
Blanking the four image corners changes the AUC by -0.0001 ± 0.0009, about as
precisely zero as an ablation can come out. That licenses "no measurable
contribution" and not "the model ignores them", which is a weaker statement and
the strongest one the measurement supports.

## What came next

Asking the model for a location directly, instead of reading one out of it,
turned out to work much better. See [03_zweiter_kopf](../03_zweiter_kopf/).

## Contents and re-running

| Path | What it is |
| --- | --- |
| `predictions_lokalisation/` | the location prior and lung map per fold, the measuring instrument itself |
| `predictions_cam_full/` | point AUC per image for every map, all 5,154 positives |

```powershell
venv\Scripts\python.exe rsna\befunde\rsna_lokalisation.py bauen --out-dir archiv\02_lokalisation\predictions_lokalisation
venv\Scripts\python.exe rsna\befunde\rsna_cam_power.py --baselines archiv\02_lokalisation\predictions_lokalisation --out-dir archiv\02_lokalisation\predictions_cam_full
```
