# Pneumonia detection on chest radiographs

A convolutional neural network that scores frontal chest radiographs for pneumonic
consolidation. It reaches an AUC of **0.845** on patients it was never trained on (five
folds, patient-grouped), and **0.885** on a different population from another continent,
after correcting for a metadata leak that inflates the raw external figure to 0.923.

Those two numbers are not the subject of this repository. The subject is what sits next to
them: what a model with no access to the image at all scores on the same data, which
shortcuts the model was found to take, which preprocessing ideas were refuted by
measurement, and why a model that discriminates well can still be unsafe to deploy.

The project was built by a freshly graduated doctor entering the clinical work life, in order to learn how these systems
behave rather than to ship a product.

> Not a medical device. Research demonstrator. No diagnostic use.

---

## Terms

Four terms carry the whole document. Readers who know them can skip ahead.

AUC (area under the ROC curve) is the probability that a randomly chosen patient with
pneumonia gets a higher score than a randomly chosen patient without it. It is the same
quantity as Harrell's c-statistic. 1.0 is perfect and 0.5 is a coin flip. The 0.5 is the
number that matters here, because it is what "knows nothing" looks like.

A confounder is something that travels with the diagnosis but is not the diagnosis. On chest
radiographs the largest one is projection: AP films are mostly taken supine with a portable
unit, at the bedside, on sicker patients. A model can learn "this looks like a portable AP
film" and score well without ever looking at the lung.

Stratification is the corresponding correction. Instead of one AUC over everything, the AUC
is computed within AP films and within PA films separately and then averaged. Inside a
projection the confounder cannot help, so what survives is closer to radiology.

Grad-CAM is a heat map showing which image regions drove the score. A plausible-looking heat
map is easy to produce and easy to over-read, so it is scored here against the
radiologist-drawn bounding boxes and against chance rather than shown as a picture on its
own.

---

## Summary

|                                           |                                                                                          |
| ----------------------------------------- | ---------------------------------------------------------------------------------------- |
| Data                                      | RSNA Pneumonia Detection Challenge: 26,684 adult chest radiographs, multi-centre, DICOM  |
| Split                                     | 22,872 development images, patient-grouped, 5 folds · 3,812 holdout images never touched |
| Stratified AUC                            | **0.845 ± 0.015** (mean ± SD across 5 folds)                                             |
| Unstratified AUC                          | 0.880 ± 0.009                                                                            |
| Header-only baseline, same stratification | **0.557**, a model that sees no pixels, only DICOM metadata (unstratified: 0.729)        |
| Margin over that baseline                 | **0.288 ± 0.011**                                                                        |
| External validation                       | **0.885** on 5,856 paediatric films from a different continent, leak-adjusted            |
| Localisation                              | Grad-CAM peak inside the annotated box 4.6× chance, but its mass only 1.6×               |
| Deployment readiness                      | **No.** Transferring the decision threshold gives NPV 0.500 (see finding 6)              |

The margin is the informative headline. 0.845 alone would flatter the model: a classifier fed
nothing but the DICOM header already reaches 0.557 on the same stratified comparison, because
sicker patients get portable AP films. What the images contribute is the difference.

That margin is also the more stable measurement. Its spread across folds is 25% smaller than
the raw stratified AUC's (0.011 against 0.015), because fold difficulty cancels out. Fold 1
looked like a better model until the pixel-blind baseline rose on that fold too. Its
validation set was simply easier.

---

## What the model does

The input is a frontal chest radiograph. RSNA ships DICOM, which is converted once to
512×512 PNG (`rsna_prepare.py`). The conversion is a faithful downscale: no contrast
equalisation, no normalisation, no crop. Every preprocessing decision belongs in the dataset
transform instead, where it can be switched per run and therefore compared. On the first
dataset CLAHE was baked into the conversion, after which preprocessing and data could no
longer be told apart.

Training reads those PNGs at 224×224 and normalises with fixed ImageNet statistics, not with
statistics computed per image (finding 4 gives the reason). Augmentation is small and
geometric: rotation up to 7°, translation up to 3%, scale 0.93 to 1.07, brightness and
contrast jitter of 0.15. Horizontal flipping is deliberately absent, because of situs
inversus and the burnt-in side markers.

The network is a ResNet-18 initialised from ImageNet weights, with the final layer replaced
by a single logit. The loss is binary cross-entropy with `pos_weight` set to the
negative-to-positive ratio of the fitting split. Optimiser AdamW, learning rate 3e-4, weight
decay 1e-4, one-cycle schedule, 8 epochs, batch size 16.

Splitting is `StratifiedGroupKFold` over patient identifiers: five outer folds over the
22,872 development images. Within the training part of each fold, a further patient-grouped
selection split chooses the checkpoint and the decision thresholds. The outer fold is
reported and never optimised against. 3,812 images are sealed off entirely.

Each run writes a metrics row to `results_rsna.csv`, per-image scores for both the reporting
and the selection split, a per-epoch history, the Grad-CAM table, and the weights of the
selected epoch. Because every score reaches disk, a follow-up question costs a re-analysis
instead of a retraining run.

Three evaluations run alongside every training run:

- the stratified AUC, computed within AP and within PA films and then averaged;
- Grad-CAM peak and mass against the radiologist-drawn boxes, each against the chance level
  given by the box area;
- a perturbation battery (blanked corners, zoom, shift, rotation, contrast and brightness
  change, blur, resolution loss) reporting how much of the score survives each manipulation.

The demo application in `webapp/` returns a probability with its uncertainty and a heat map.
It never returns a binary label; finding 6 is the reason.

---

## The task, and the one decision that shapes it

RSNA labels each film as `Normal`, `Lung Opacity` (infiltrate, with a radiologist-drawn box),
or `No Lung Opacity / Not Normal`, meaning abnormal but not pneumonia. That middle class is
~11,800 of the 26,684 images, and its treatment decides what the model learns.

> Here: `Lung Opacity` = 1. `Normal` and the middle class = 0.

Clinically the question is "pneumonia, yes or no", not "ill, yes or no". Dropping the middle
class answers the second question while claiming to have answered the first.

The choice is also measurable. Removing the middle class pushes the header-only confounder
from 0.729 to 0.824. The apparently cleaner task is the more biased one, and the reason is in
the cross-tabulation: 48.2% of AP films are middle class. Those are sick patients at the
bedside without pneumonia, the counterexamples that stop "AP" from collapsing into
"pneumonia". Delete them and projection alone predicts opacity in 74% of AP films. Choosing
the easier task buys a shortcut worth 0.095 AUC.

---

## Findings

### 1. In the first dataset, the file dimensions gave the answer away

The project started on the Kermany paediatric dataset. The model classified well and Grad-CAM
never pointed at anything. The reason:

> The JPEG dimensions alone separate the classes at AUC 0.915. In the official test folder,
> image width alone reaches 0.950, beating the CNN's 0.942.

This is not a bug in the code. NORMAL films carry systematically about 2.4× the pixel count
of PNEUMONIA films (median 1654×1323 against 1160×776, so roughly 1.4× per side). Probably
different acquisition setups, devices or age groups, though the dataset does not say and it
could not be verified. Anything derivable from that size difference (texture sharpness,
aspect ratio, even the shape of a lung mask) is a route to the right answer that has nothing
to do with the chest.

This is why the project moved to RSNA, where every image is 1024×1024 and the confounders are
written in the DICOM header, so they can be measured instead of reconstructed.

### 2. The confounder that remained, and why it was stratified rather than hidden

On RSNA the header-only classifier reaches 0.729, and it is almost entirely projection:

| Header feature         | AUC → pneumonia |
| ---------------------- | --------------- |
| `ViewPosition` (AP/PA) | 0.706           |
| age                    | 0.530           |
| pixel spacing          | 0.517           |
| sex                    | 0.510           |
| all combined           | 0.729           |

Within a single projection, the header-only score collapses to ~0.556 on the full set, and to
0.557 averaged over the five validation folds, which is the figure used as the baseline above.
The confounder is therefore binary and exactly known, which calls for stratification rather
than matching. Matching on a continuous nuisance variable (as the first dataset required)
discarded two thirds of the data; stratifying on a binary one does not cost a single image.

+0.035 ± 0.006 of the headline number is projection (0.880 unstratified against 0.845
stratified; in fold 0 alone it reaches 0.044). That is the difference between the number that
could have been published and the number reported here.

### 3. The model reads the lung, measurably but diffusely

RSNA provides radiologist-drawn boxes, so the heat map can be scored instead of admired.

|                            | measured      | chance        | ratio |
| -------------------------- | ------------- | ------------- | ----- |
| Grad-CAM peak inside a box | 0.539 ± 0.036 | 0.117 ± 0.005 | 4.6×  |
| Grad-CAM mass inside a box | 0.192 ± 0.013 | 0.117 ± 0.005 | 1.6×  |

The chance baseline is the fraction of image area the boxes cover, and it has to be reported.
A hit rate of 0.6 sounds impressive and would be nearly nothing if the boxes covered 55% of
the image.

The reading the data supports: the peak lands on the pathology in just over half of cases
(4.6× chance, but still only half), and the rest of the map is diffuse. That is a weaker and
more accurate claim than "Grad-CAM looks plausible", which is what a gallery of cherry-picked
heat maps would have supported. Both numbers replicated across five folds.

Related: RSNA films carry burnt-in markers. Ablating the image corners changes the AUC by
-0.0001 ± 0.0009, about as precisely zero as an ablation can come out. No measurable
contribution, which is a weaker statement than "the model ignores them", and the strongest one
the measurement supports.

### 4. The confounder cannot be removed from the images

Three routes were tried and all three were refuted: pixel-exact lung masking, a rectangular
lung crop, and per-image intensity normalisation. Each was the plan at some point, and each
refutation turned out to be a more useful result than the plan would have been.

Lung segmentation as preprocessing does not help here; it actively hurts. All three
mechanisms by which it could have worked were refuted individually, which saved 11.5
hours of compute that had already been scheduled. The decisive argument came from looking at
the alternative: a rectangle around the lungs necessarily contains the mediastinum, but
pixel-exact masking does not remove the mediastinum either. It turns it into a hole, and the
shape of that hole is the cardiac and vascular silhouette. Masking may therefore not delete
the confounder at all. It may re-encode it as a contour, and a contour is the easier feature
for a convolutional network to read, not the harder one.

Measured on 22,846 images, using three clinically readable properties of the cutout
(cardiothoracic ratio, area, vertical position). No grey value enters:

|                                           | AUC   | for comparison                     |
| ----------------------------------------- | ----- | ---------------------------------- |
| Cutout silhouette → projection            | 0.692 | framing of the rectangle: 0.714    |
| Cutout silhouette → pneumonia, stratified | 0.593 | rectangle's crop parameters: 0.551 |

Separately, masking would delete 12.9% of the annotated box area (7.8% into the mediastinal
cutout, 5.1% beyond the lung outline altogether). So masking keeps about 90% of the
above-chance projection signal (0.692 - 0.5 against 0.714 - 0.5), builds a stronger new
shortcut than cropping does, and deletes an eighth of the finding. That eighth is concentrated
rather than spread out: in 9.2% of positive films more than a fifth of the box lies in the
mediastinal region. Those are the retrocardiac and paramediastinal consolidations.

The gap between the two shortcut figures, silhouette 0.593 against crop parameters 0.551, is
+0.042, bootstrap 95% CI [0.030, 0.051], so it is not a rounding artefact. These are three
hand-picked summary numbers; a network would see the whole contour, so 0.692 and 0.593 are
lower bounds.

The rectangular crop was the second attempt. A square bounding box around the lung mask plus
margin, precomputed rather than applied at run time, was evaluated over five folds at the
same 8-epoch budget, paired within fold. The geometry checks done before training were favourable: the crop
preserves 99.7% of annotated box area, and its parameters predict the class at 0.552
stratified, so it builds no obvious shortcut of its own.

The pre-registered primary endpoint was AUC(model score → `ViewPosition`), 0.8166 ± 0.0098,
and it was to fall. It rose in all five folds, mean +0.027 (+0.011, +0.041, +0.004, +0.070,
+0.010). Stratified and overall AUC did not move (+0.003 and +0.001).

The mechanism is measurable. The side length of the crop window alone predicts AP against PA
at AUC 0.685, against 0.714 for the framing as a whole. The adaptive crop takes almost the
entire channel it was meant to close and writes it back into the image as a global
magnification factor: AP films are enlarged 1.197× on average, PA films 1.117×. Texture
frequency is easier for a convolutional network to read than framing, so the intervention made
the endpoint worse rather than better.

The margin is not the culprit, which was the obvious next suspicion:

| margin                 | window side (median) | AUC(side → AP/PA) | box area below 90% preserved |
| ---------------------- | -------------------- | ----------------- | ---------------------------- |
| 0.05 (as run)          | 0.881                | 0.685             | 0.7%                         |
| 0.02                   | 0.833                | 0.686             | 1.1%                         |
| 0.00                   | 0.801                | 0.686             | 1.7%                         |
| no dilation, no margin | 0.738                | 0.683             | 5.9%                         |

Cropping tighter costs pathology and gains nothing at the confounder. Percentile rectangles
(p1/p99 instead of min/max) are worse still, 0.704 and 0.711, because they couple window size
more faithfully to true lung size, and lung size is the signal.

The only neutral crop keeps the window size constant and takes only the position from the
mask:

| fixed side       | AUC(geometry → AP/PA) | box area preserved | below 90% |
| ---------------- | --------------------- | ------------------ | --------- |
| adaptive, as run | 0.714                 | 0.997              | 0.7%      |
| 0.85             | 0.561                 | 0.998              | 0.5%      |
| 0.80             | 0.554                 | 0.996              | 1.4%      |
| 0.75             | 0.550                 | 0.989              | 3.2%      |
| 0.70             | 0.549                 | 0.974              | 8.7%      |
| 0.65             | 0.548                 | 0.942              | 22.4%     |

Below a side of 0.75 the box preservation collapses. Neutral, however, is not better: a fixed
side merely returns the channel to the level it has without any crop.

The underlying reason is projective geometry rather than an artefact. Supine AP is taken at
roughly 100 cm focus-film distance against roughly 180 cm for PA, so apparent lung size
differs by projection, and no anatomical reference length is projection-independent. Every
scale normalisation normalises the confounder along with it.

Per-image normalisation, the third route, behaves the same way. Measured on 96 images
(48 AP, 48 PA, noise about ±0.05), four intensity features taken inside the eroded lung mask
predict AP against PA at 0.721 under the current fixed ImageNet normalisation, at 0.768 after
per-image z-normalisation, and at 0.818 after CLAHE. Both supposed improvements make the
channel stronger, because the normalisation statistic is computed from the image, and how much
abdomen, shoulder and black border the image contains depends on the projection.

A related trap is worth recording. Eight trivial whole-image statistics predict AP against PA
at AUC 0.939, more than the model itself manages at 0.8166, with Laplace variance (sharpness)
alone at 0.844. The same measurements inside the eroded lung mask give 0.561 for sharpness and
0.572 for noise, which is close to nothing. The 0.844 comes from collimation edges, hardware,
bed frame and border lettering on the supine films, not from lung markings. Any whole-image
statistic on a radiograph measures the border first.

The pattern across all three routes is the same, and it is now measured three times: a
deterministic transform can re-encode information, it cannot delete it. Deleting requires
either added noise (augmentation) or a constraint applied during training.

A fourth prediction was refuted the same way. The Kermany aspect ratio differs strongly by class
(mean 1.25 for normal films against 1.51 for pneumonia) and the pipeline stretches everything
to a square, so class-correlated distortion was expected to hurt on external validation.
Padding instead of stretching, over 5,856 images: -0.0001. Measured, discarded.

### 5. External validation: the discrimination transfers

Five RSNA-trained checkpoints, pure inference, no fine-tuning, on 5,856 Kermany images (3,054
patient groups). Every axis shifts at once: adults → children aged 1 to 5, USA → Guangzhou,
1024² DICOM → JPEG of varying size, prevalence 0.225 → 0.730.

|                                                        | AUC                                                             |
| ------------------------------------------------------ | --------------------------------------------------------------- |
| Raw ensemble, 5 checkpoints                            | 0.923 [0.916 to 0.930], bootstrap grouped by patient            |
| Single folds                                           | 0.886 ± 0.019, so ensembling is worth more here than internally |
| Metadata leak alone (the 0.915 problem, still present) | 0.914                                                           |
| Leak-adjusted                                          | **0.885**                                                       |
| Internal comparison (RSNA, stratified)                 | 0.845 ± 0.015                                                   |

The ranking transfers. What it does not license is a clean "better than internal" claim: the
external task has no middle class (Kermany is normal against pneumonia only), which this
project argues elsewhere is the easier and more biased framing, and prevalence differs
threefold. The external number reads as "no collapse across a hard domain gap", not as an
improvement.

One check is worth more than the headline: the model score and the leak are largely
complementary channels. Model alone 0.923, dimensions alone 0.914, both together 0.966. The
model adds ~0.05 on top of the leak, and the leak ~0.04 on top of the model. If the model were
simply re-reading the file dimensions, that first increment would not exist. Complementary is
what is measured; strict independence is not, and additivity alone would not establish it.

One error in this analysis changed the number and is recorded for that reason. The
leak-stratified AUC was first weighted by stratum size. But one stratum held over a thousand
positives against a single negative. Its whole AUC rested on comparisons with that one image,
and size-weighting gave that noise full weight (≈0.85). Weighting instead by discordant pairs,
which is the actual information content of a stratum, gives 0.885. A regression test now
reconstructs the case. The corrected figure is the robust one: it moves by less than 0.002
across every stratum definition tried, while the size-weighted variant swings by more than
0.03.

### 6. The calibration does not transfer, and clinically that is the decisive part

Carrying the operating threshold (0.483) across, unchanged:

|             |           |
| ----------- | --------- |
| Sensitivity | 0.649     |
| Specificity | 0.950     |
| PPV         | 0.972     |
| NPV         | **0.500** |

> Of the cases this model calls negative, half actually have pneumonia.

The ranking transferred; the threshold did not. Prevalence went from 0.225 to 0.730 and the
score distribution shifted with it (sensitivity fell from ~0.82 internally to 0.649, which
prevalence alone cannot explain). This is the distinction that usually disappears between
"AUC 0.92" and "ready to use", and it is why the demo application reports a probability with
its uncertainty and never a binary label.

The same problem exists within RSNA, between projections. In one fold, a single shared
threshold gave sensitivity 0.894 in AP films and 0.677 in PA, one number behaving as two
different tests. Prevalence differs fourfold between the two strata (0.383 against 0.093), the
score distributions differ with it, so one shared cut-off sits in a different place in each.

Setting the threshold per projection (on a separate selection split, never on the reporting
set) narrows the sensitivity gap from 0.299 ± 0.073 to 0.067 ± 0.038. Those are means over the
four folds that carry per-projection thresholds; fold 0 predates the change.

---

## Method rules

- Splits are patient-grouped. No patient appears in both training and validation, and this
  was checked rather than assumed.
- The holdout is still sealed. 3,812 images, untouched, to be evaluated exactly once at the
  end. Using it for an intermediate check would quietly burn the only unbiased estimate in
  the project.
- Selection and reporting use separate sets. The checkpoint is chosen on an inner
  patient-grouped selection split; the outer validation fold is only reported. Thresholds come
  from the selection split. Optimistic "oracle" variants run alongside so the gap stays
  visible.
- Comparisons are paired within a fold, because fold difficulty varies more than the effects
  being measured. An unpaired difference of 0.005 means nothing.
- Every number sits next to its null hypothesis: header-only baseline, chance box coverage,
  leak-only AUC. Where no null could be constructed, that is stated.
- Endpoints are written down before the run. Each intervention carries its primary endpoint,
  its expected effect size and its "this would refute the idea" case in the source file, dated
  before the first result existed. The crop in finding 4 is what that rule is for: the
  endpoint moved the wrong way, and the pre-registration is what makes that readable as a
  result rather than as a failed run.
- Predictions are saved. Every run writes per-image scores, so re-analysis costs seconds
  instead of another nine-hour run. That is what the first re-analysis on the earlier dataset
  actually cost, before this rule existed.

---

## Current experiment: reweighting instead of removing

Since no deterministic transform removes the projection channel, the current experiment
addresses the incentive instead of the evidence. The model reads the projection not because it
is visible but because it is useful: `ViewPosition` → label has AUC 0.706, so a model trained
on label accuracy has every reason to encode it. The evidence itself is spread over heart
size, scapulae, diaphragm position, framing and sharpness at once, which is why removing any
single carrier changed nothing.

The imbalance sits in the cross-tabulation of the development set:

|     | label 0 | label 1 | prevalence |
| --- | ------- | ------- | ---------- |
| AP  | 6,436   | 3,998   | 0.383      |
| PA  | 11,282  | 1,156   | 0.093      |

`--balance-view` draws each training image with the weight a chi-square test is built on, the
count expected under independence divided by the count observed:

    w(v, y) = n_v * n_y / (N * n_vy)

That is 1.26 for AP-negative, 0.59 for AP-positive, 0.85 for PA-negative and 2.42 for
PA-positive. Both marginals stay exactly as they were, the overall prevalence of 0.225 and the
AP to PA ratio; only the association between them is cut. That is deliberate, because it
leaves `pos_weight` valid and the class imbalance is not corrected twice. The weights come
from the fitting split of the current fold alone; selection and reporting splits are never
reweighted. The price at full dose is 14% of the effective sample size by the Kish measure,
19,698 of 22,872.

Pre-registered before the first run: the primary endpoint AUC(model score → `ViewPosition`)
must fall from 0.8166 ± 0.0098; the stratified AUC must not fall from 0.8449 ± 0.0147; the raw
AUC is expected to fall, and that is the success rather than a regression, because the 0.880
contains the projection contribution.

A dose-response curve on fold 0, with the weight raised to the power α, fixed the operating
point:

| α   | `ViewPosition` → label in the training stream | score → projection | stratified AUC | raw AUC | Grad-CAM peak | effective n |
| --- | --------------------------------------------- | ------------------ | -------------- | ------- | ------------- | ----------- |
| 0.0 | 0.706                                         | 0.8076             | 0.8209         | 0.8649  | 0.527         | 15,248      |
| 0.5 | 0.611                                         | 0.7671             | 0.8194         | 0.8562  | 0.444         | 14,762      |
| 1.0 | 0.500                                         | 0.7381             | 0.8111         | 0.8434  | 0.340         | 13,133      |

The shape of that curve is the finding. At 46% of the dose, 58% of the effect arrives against
44% of the cost. The benefit is concave and the harm convex, so the operating point lies in
the middle rather than at the maximum: 0.490 endpoint gain per unit of Grad-CAM loss at
α = 0.5, against 0.372 at α = 1.0. Split by projection over the same 300 images, the Grad-CAM
peak runs 0.596 → 0.491 → 0.342 in AP films (4.7× → 3.8× → 2.7× chance) and 0.306 → 0.296 →
0.333 in PA films (4.7× → 4.6× → 5.2×). The loss sits in AP, where the positives are weighted
down; the gain in PA arrives only at full dose.

Interim, and explicitly interim: two of the five folds are complete at α = 0.5.

| difference against baseline  | fold 0  | fold 1  | mean of 2 folds |
| ---------------------------- | ------- | ------- | --------------- |
| score → projection (primary) | -0.0405 | -0.0354 | -0.0379         |
| stratified AUC (secondary)   | -0.0015 | -0.0191 | -0.0103         |
| raw AUC                      |         |         | -0.0132         |
| Grad-CAM peak                | -0.083  | -0.072  | -0.0773         |

The primary endpoint is moving in the pre-registered direction and is stable so far (spread
0.0036 over two folds). The secondary condition is the one at risk: the pre-set tolerance for
the stratified AUC is -0.015 and the current mean is -0.0103, carried by a single fold. Folds
2 to 4 decide this point, not the primary endpoint. Two folds are also too few for an
interval, so these numbers are a status report and not a result.

One side finding replicates already: the per-projection sensitivity gap is 0.016 in fold 1 and
0.005 in fold 0, against 0.072 in the corresponding baseline.

---

## Repository

The scripts are named in the order they run.

```
Data and the null hypothesis
  rsna_prepare.py             DICOM to PNG, once
  rsna_splits.py              patient-grouped splits, stratified, holdout sealed off
  rsna_data.py                DICOM header extraction
  rsna_metadata_leak_check.py the header-only baseline, run before any training

Training and evaluation
  rsna_train.py               training, stratified metrics, Grad-CAM against boxes,
                              perturbation battery, per-epoch history, --balance-view
  rsna_gradcam_grid.py        heat maps for visual inspection, hits and misses separated

Does segmentation help? (answer: no, and it hurts)
  rsna_make_masks.py          U-Net lung masks, resumable, raw output cached
  rsna_cam_lung_check.py      does the heat-map peak fall outside the lung?
  rsna_mask_sweep.py          is the apparent headroom just a mask artefact?
  rsna_mask_silhouette.py     what pixel-exact masking would really do

The rectangular crop (answer: it re-encodes the confounder)
  rsna_crop_geometry.py       can a crop reduce the projection confounder at all?
  rsna_make_crops.py          precomputed crop, bounding boxes rewritten to match
  rsna_crop_compare.py        paired comparison, endpoints fixed in the module header
  rsna_crop_qc.py             visual check of the crop windows

External validation
  rsna_external_kermany.py    leak stratification, grouped bootstrap, threshold transfer
  splits.py                   patient grouping for that dataset

tests/                        six suites, run against hand-computed cases
archiv/kermany/               the first phase, closed but kept: it is the evidence
webapp/                       demo interface
```

Every script's module docstring says what it produces, why it exists, and how to read its
result, including which value is the null and what would falsify the claim.

```bash
# reproduce the headline
python rsna/befunde/rsna_metadata_leak_check.py          # the null hypothesis, first
python rsna/pipeline/rsna_splits.py
python rsna/pipeline/rsna_prepare.py
for f in 0 1 2 3 4; do python rsna/pipeline/rsna_train.py --fold $f --device directml; done
python rsna/befunde/rsna_external_kermany.py
```

---

## Hardware, and why it appears in the README

A Radeon RX 5500 XT. RDNA1, so no CUDA and no ROCm, which leaves `torch-directml` over
DirectX 12 as the route. No mixed precision, batch size 16, DataLoader workers disabled on
Windows.

It appears here because it shaped the design. Every hour of compute had to be justified before
it was spent, which is why this project measures whether an idea can work before training on
it. The segmentation question was settled by three cheap measurements instead of an 11.5-hour
run. Having to argue for each run produced better planning than an unlimited budget would
have.

---

## Limitations

- Single architecture (ResNet-18, ImageNet-initialised). No architecture search, which would
  not have answered any question this project asks.
- The holdout is still sealed, so the final unbiased number does not exist yet.
- External validation is a single dataset, and a paediatric one. It is a hard domain gap, not
  a representative one.
- The Grad-CAM mass ratio of 1.6× is weak. The map localises with its peak and is diffuse
  elsewhere, and that has not been fixed.
- The reweighting experiment costs localisation quality: the Grad-CAM peak falls from 0.527 to
  0.444 at the chosen operating point. Whether that trade is acceptable is not settled by the
  numbers alone.
- Radiologist-drawn boxes are the reference standard, not microbiology or follow-up. The model
  is measured against reader opinion, with all that implies.
- Everything here is the work of one person learning. Errors should be assumed to remain; the
  tests and the saved predictions exist so that they can be found.
