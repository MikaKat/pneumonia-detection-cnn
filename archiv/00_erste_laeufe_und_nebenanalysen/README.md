# Baseline runs, and the three preprocessing routes that were refuted

Two things live here. The five-fold baseline that every later comparison is
measured against, and a collection of cheap side analyses that decided, without
a single training run, that three plausible preprocessing ideas were not worth
training on.

## The baseline

Five patient-grouped folds, ResNet-18, 8 epochs, no second head, no reweighting.

| | value |
| --- | --- |
| stratified AUC | 0.8449 ± 0.0147 |
| unstratified AUC | 0.880 ± 0.009 |
| score to projection | 0.8166 ± 0.0098 |
| header-only baseline, stratified | 0.557 |
| header-only baseline, unstratified | 0.729 |

The gap between 0.880 and 0.845 is the projection contribution, +0.035 ± 0.006.
The gap between 0.845 and 0.557 is what the pixels add.

`predictions_rsna_base/` holds the run that all four later arms are anchored
against. `predictions_rsna/` is older and is the collection bucket of the first
RSNA period.

## The three routes, all refuted before training

The pattern across all three is the same, and it was measured three times: a
deterministic transform can re-encode information, it cannot delete it.

**Pixel-exact lung masking.** Masking does not remove the mediastinum, it turns
it into a hole, and the shape of that hole is the cardiac silhouette. Measured
on 22,846 images with three clinically readable properties of the cutout, no
grey value entering: the silhouette alone predicts the projection at 0.692,
against 0.714 for the framing of the whole rectangle. So masking keeps about 90
percent of the above-chance projection signal and builds a new shortcut of 0.593
in the process. It would also delete 12.9 percent of the annotated box area,
concentrated rather than spread: in 9.2 percent of positive films more than a
fifth of the box lies in the mediastinal region, which is where the retrocardiac
and paramediastinal consolidations are.

**A whole-image statistic measures the border first.** Eight trivial statistics
predict AP against PA at 0.939, more than the model itself manages, with Laplace
variance alone at 0.844. The same measurements inside the eroded lung mask give
0.561 and 0.572. The 0.844 comes from collimation edges, bed frame and border
lettering on the supine films, not from lung markings.

**Per-image intensity normalisation.** Four intensity features inside the eroded
lung mask predict the projection at 0.721 under the fixed ImageNet
normalisation, at 0.768 after per-image z-normalisation and at 0.818 after
CLAHE. Both supposed improvements make the channel stronger, because the
normalisation statistic is computed from the image, and how much abdomen,
shoulder and black border an image contains depends on the projection.

These three measurements together saved 11.5 hours of compute that had already
been scheduled.

## The external validation lives here too

`predictions_rsna/external_kermany.csv` is the per-image output of the external
run on 5,856 paediatric films. Headline numbers and the weighting error that was
found and corrected are in the main README, finding 5.

## Contents

| Path | What it is |
| --- | --- |
| `predictions_rsna_base/` | the five-fold baseline, the anchor for phases 6 to 9 |
| `predictions_rsna/` | first RSNA period plus the side analyses below |
| `predictions_rsna/mask_silhouette.csv` | what pixel-exact masking would really do |
| `predictions_rsna/mask_sweep.csv` | is the apparent headroom just a mask artefact? |
| `predictions_rsna/cam_lung.csv` | does the heat-map peak fall outside the lung? |
| `predictions_rsna/external_kermany.csv` | the external validation, per image |
| `predictions_rsna/crop_*.csv` | crop geometry, see [07_zuschnitt](../07_zuschnitt/) |
| `predictions_rsna/restkanal.csv` | what the fixed crop leaves, see [07_zuschnitt](../07_zuschnitt/) |
| `rsna_mask_qc*.png` | visual check of the segmenter output |

## Re-running

```powershell
venv\Scripts\python.exe rsna\befunde\rsna_mask_silhouette.py --out archiv\00_erste_laeufe_und_nebenanalysen\predictions_rsna\mask_silhouette.csv
venv\Scripts\python.exe rsna\befunde\rsna_external_kermany.py --out archiv\00_erste_laeufe_und_nebenanalysen\predictions_rsna\external_kermany.csv
```
