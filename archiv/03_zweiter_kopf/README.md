# Asking the model where, instead of reading it out afterwards

This is the experiment that produced the architecture in the main directory. It
is the only one on this list that ended up in the shipped model.

## What was built

A second output on the same trunk: a 14 by 14 field, one 1 by 1 convolution on
the `layer3` features, supervised against the radiologist boxes with a weight
measured from the first training batch rather than chosen. Only films that carry
a box contribute to that loss.

The head taps `layer3` and not `layer4` for two reasons, and the second is the
one that mattered later. At 224 pixels `layer3` is already 14 by 14, so nothing
in the trunk has to change and the two arms differ in the head and in nothing
else. And pooling to a fixed grid means the head keeps its 14 by 14 whatever the
input size, so the measuring stick stopped moving with the thing it measures.
That is what made [08_aufloesung](../08_aufloesung/) readable at all.

## Two arms, because one question was open

`exclude` trains the head only on films that have a box. `empty` also shows it
healthy films with an empty field, so it can learn to stay quiet. Which is
better was not obvious in advance and was decided by measurement.

## Results, all 5,154 positive validation images

| | point AUC in the lung | hit rate | degenerate maps |
| --- | --- | --- | --- |
| Head field, `exclude` | 0.9123 | | |
| Head field, `empty` | 0.8996 | | |
| Location prior | 0.7520 | 0.5714 | |
| Grad-CAM, `exclude` | 0.7312 | 0.5982 | 119 |
| Grad-CAM, `empty` | 0.7292 | 0.6059 | 253 |
| Lung map | 0.7011 | 0.5326 | |
| Grad-CAM, no head | 0.6786 | 0.4212 | 108 |

Three readings, in decreasing order of confidence.

The head is the first map in this project that clearly beats the anatomical
prior, and it stays ahead in every typicality quintile (0.8753 in the most
atypical, 0.9355 in the most typical) while the prior runs from 0.5124 to
0.9208.

The Grad-CAM of the same networks improves as well, although nothing was changed
about how it is computed. Supervision at the head reaches back into the shared
trunk: +0.1753 [+0.1507, +0.2000] on the hit rate for `exclude`, paired per
image.

But the improved Grad-CAM only reaches the prior, it does not beat it: -0.0208
(t = -1.27) and -0.0228 (t = -1.88). Unclear, and unclear here means measured too
imprecisely. Five folds resolve about 0.04 on this quantity.

The sentence for the portfolio is therefore not "the head improves Grad-CAM" but
this: Grad-CAM stops being worse than anatomy, and the location comes from the
head. 0.9123 against 0.7312 is not a gradual difference.

## What it costs, and what it does not do

Adding the head costs nothing on the diagnosis: +0.0081 [+0.0014, +0.0149]
stratified AUC, non-inferiority at a margin of 0.01 fixed beforehand. The honest
sentence is "costs nothing", not "helps", because superiority was never
registered. `exclude` won the localisation endpoint in all five folds and is the
arm that ships.

It does not lower the projection confounder: +0.0061 [-0.0108, +0.0229]. For the
confounder the rise is the problem, so the upper end decides, and for `exclude`
it sits above the margin. That is why decoupling stayed the job of
[04_umgewichtung](../04_umgewichtung/).

## The head fires on healthy films, and that is why the app draws no box

Follow-up run on films without pneumonia. The head is not a pure opacity
detector, which was the gate and it passed (stratified 0.767). But its level is
uncalibrated: **it lights up somewhere on 62 percent of normal films.**

Scored as a detection task with the competition metric it reaches 0.1362 alone
and 0.1556 chained behind the classifier, against 0.0249 for the location prior.
Almost half the loss is false alarms. The 14 by 14 grid is not the limit: a
perfect answer on that grid would score 0.8111, so coarseness is not the binding
constraint.

This is the direct reason the web app draws the field as a gradient with no box,
no outline and no cut-off at 0.5. Any of those would be a claim about exactly
the quantity that was measured and found wanting.

## One fold was trained with a broken weight

`p5_ex_f1_lambda_kaputt/` is fold 1 of the `exclude` arm with a localisation
weight that was computed on the first batch only and came out wrong. It was
found, the fold was retrained, and both the broken and the repaired run are kept
because the pair is informative: nearly identical classification (0.8431 against
0.8426) and the lowest and the highest confounder value of the whole arm. The
mixture of the two losses shifts mostly what the model looks at, less how well
it separates.

## Contents and re-running

| Path | What it is |
| --- | --- |
| `predictions_p5_ref/` | the same recipe without the head, the comparison partner |
| `predictions_p5_head_em/` | the `empty` arm |
| `predictions_p5_auswertung/` | the paired comparison and both follow-up runs |
| `predictions_cam_p5/` | the three-map table above, per image |
| `predictions_kopf/` | the grid ceiling measurement |
| `p5_ex_f1_lambda_kaputt/` | the fold with the broken weight |

The winning arm itself, `predictions_final_model/`, stayed in the main directory:
it is where the shipped calibration curves and threshold come from.

```powershell
venv\Scripts\python.exe rsna\befunde\rsna_phase5_auswertung.py bericht --out-dir archiv\03_zweiter_kopf\predictions_p5_auswertung
venv\Scripts\python.exe rsna\befunde\rsna_phase5b_falschalarm.py --out-dir archiv\03_zweiter_kopf\predictions_p5_auswertung
venv\Scripts\python.exe rsna\befunde\rsna_phase5b_detektion.py --out-dir archiv\03_zweiter_kopf\predictions_p5_auswertung
```
