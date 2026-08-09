# Cropping to the lungs

Two attempts, several years of Kaggle folklore, and one side finding that turned
out to be the most informative thing in the folder.

## Attempt one: the adaptive crop, refuted

A square bounding box around the lung mask plus a margin, precomputed rather
than applied at run time, five folds, paired within fold. The geometry checks
before training were favourable: the crop preserves 99.7 percent of annotated
box area, and its own parameters predict the class at only 0.552 stratified, so
it builds no obvious shortcut.

The pre-registered primary endpoint was that `AUC(score to ViewPosition)` falls
from 0.8166 ± 0.0098. It **rose in all five folds**, mean +0.027 (+0.011, +0.041,
+0.004, +0.070, +0.010). Stratified and overall AUC did not move.

The mechanism is measurable. The side length of the crop window alone predicts
AP against PA at 0.685, against 0.714 for the framing as a whole. The adaptive
crop takes almost the entire channel it was meant to close and writes it back
into the image as a global magnification factor: AP films are enlarged 1.197x on
average, PA films 1.117x. Texture frequency is easier for a convolutional
network to read than framing, so the intervention made the endpoint worse.

The margin is not the culprit, which was the obvious next suspicion:

| margin | window side (median) | AUC(side to AP/PA) | box area below 90% preserved |
| --- | --- | --- | --- |
| 0.05 (as run) | 0.881 | 0.685 | 0.7% |
| 0.02 | 0.833 | 0.686 | 1.1% |
| 0.00 | 0.801 | 0.686 | 1.7% |
| no dilation, no margin | 0.738 | 0.683 | 5.9% |

Percentile rectangles are worse still, 0.704 and 0.711, because they couple
window size more faithfully to true lung size, and lung size is the signal.

The underlying reason is projective geometry rather than an artefact. Supine AP
is taken at roughly 100 cm focus-film distance against roughly 180 cm for PA, so
apparent lung size differs by projection, and no anatomical reference length is
projection-independent. Every scale normalisation normalises the confounder
along with it.

## Attempt two: the fixed-size crop, also refuted

The only neutral crop keeps the window size constant and takes only the position
from the mask:

| fixed side | AUC(geometry to AP/PA) | box area preserved | below 90% |
| --- | --- | --- | --- |
| adaptive, as run | 0.714 | 0.997 | 0.7% |
| 0.85 | 0.561 | 0.998 | 0.5% |
| 0.80 | 0.554 | 0.996 | 1.4% |
| 0.75 | 0.550 | 0.989 | 3.2% |
| 0.70 | 0.549 | 0.974 | 8.7% |
| 0.65 | 0.548 | 0.942 | 22.4% |

Side 0.80 was chosen: below 0.75 the box preservation collapses. Run as a full
pre-registered five-fold arm against the anchor of
[03_zweiter_kopf](../03_zweiter_kopf/):

| | anchor | this arm | paired difference | |
| --- | --- | --- | --- | --- |
| score to projection | 0.7467 | 0.7566 | +0.0099 [-0.0147, +0.0345] | failed |
| stratified AUC | 0.8368 | 0.8452 | +0.0084 | guard held |

The point estimate moves the wrong way again. Interval half-width 0.0246,
against 0.025 predicted before the run. The rise in stratified AUC is most
likely a resolution effect: cropping and rescaling to 224 gives the lungs more
pixels.

## The side finding, which is worth more than the verdict

If a neutral crop removes the framing, what is left? `rsna_restkanal.py`
measures what the model can actually see of the geometry, before and after.

The window geometry itself falls by 77 percent. But what the model sees falls
only from 0.2610 to **0.1638**, a drop of 37 percent. So the crop relieves the
framing and leaves the anatomy.

And the remainder is almost entirely horizontal: width carries 0.1481, height
only 0.0304. The channel sits on thoracic width, which is itself partly a
finding rather than an artefact. That is a much harder thing to remove than a
window edge, and it is the reason the two remaining image interventions were
tried on other axes.

## Contents and re-running

| Path | What it is |
| --- | --- |
| `predictions_rsna_crop/` | the adaptive crop, five folds |
| `predictions_p7_fix080/` | the fixed-size crop at side 0.80, five folds |
| `predictions_lokalisation_fix080/` | the location prior rebuilt in crop geometry, because the old one no longer fits the frame |
| `results_rsna_crop.csv` | the metrics rows of the crop runs |
| `crop_qc*.png`, `crop_varianten*` | visual check of the crop windows and the variant table |

```powershell
venv\Scripts\python.exe rsna\befunde\rsna_phase7_auswertung.py --pred-dir archiv\07_zuschnitt\predictions_p7_fix080
venv\Scripts\python.exe rsna\befunde\rsna_restkanal.py --out archiv\00_erste_laeufe_und_nebenanalysen\predictions_rsna\restkanal.csv
```
