# Pneumonia detection on chest radiographs, and how hard I tried to disprove it

A convolutional neural network that flags pneumonic consolidation on frontal chest
radiographs. It reaches an AUC of **0.845** on patients it was never trained on (five-fold,
patient-grouped), and **0.885** on a completely different population from another continent,
after correcting for a metadata leak that inflates the raw figure to 0.923.

Those two numbers are not the point of this repository. The point is what sits next to them:
what a model with *no access to the image at all* scores on the same data, which shortcuts
the model was caught taking, which of my own ideas turned out to be wrong, and why a model
that discriminates well can still be unsafe to deploy. A portfolio that shows a good number
is common. One that shows how the number was defended against its author's own doubts is
rarer, and I think more useful.

I am a physician heading into radiology. I built this to learn how these systems actually
behave, not to ship a product.

> **Not a medical device.** Research demonstrator. No diagnostic use.

---

## How to read the numbers

Four terms carry the whole document. If you already know them, skip ahead.

AUC (area under the ROC curve) is the probability that a randomly chosen patient *with*
pneumonia gets a higher score from the model than a randomly chosen patient without it. It
is the same quantity as Harrell's c-statistic. 1.0 is perfect and 0.5 is a coin flip. The 0.5
is the number that matters here, because it is what "knows nothing" looks like.

A confounder is something that travels with the diagnosis but is not the diagnosis. On chest
radiographs the big one is projection: AP films are mostly taken supine with a portable unit,
at the bedside, on sicker patients. A model can learn "this looks like a portable AP film"
and score well without ever looking at the lung.

Stratification is the fix. Instead of one AUC over everything, report the AUC *within* AP
films and *within* PA films separately, then average. Inside a projection the confounder
cannot help, so what survives is closer to radiology.

Grad-CAM is a heat map showing which image regions drove the score. It is easy to produce a
plausible-looking heat map and easy to over-read one, so here it is scored against the
radiologist-drawn bounding boxes and against chance rather than shown as a pretty picture on
its own.

---

## The one-minute version

| | |
| --- | --- |
| Data | RSNA Pneumonia Detection Challenge: 26,684 adult chest radiographs, multi-centre, DICOM |
| Split | 22,872 development images, patient-grouped, 5 folds · 3,812 holdout images never touched |
| Stratified AUC | **0.845 ± 0.015** (mean ± SD across 5 folds) |
| Unstratified AUC | 0.880 ± 0.009 |
| Header-only baseline, same stratification | **0.557**, a model that sees *no pixels*, only DICOM metadata (unstratified: 0.729) |
| Margin over that baseline | **0.288 ± 0.011** |
| External validation | **0.885** on 5,856 paediatric films from a different continent, leak-adjusted |
| Localisation | Grad-CAM peak inside the annotated box 4.6× chance, but its mass only 1.6× |
| Deployment readiness | **No.** Transferring the decision threshold gives NPV 0.500 (see below) |

The margin is the honest headline. `0.845` alone would flatter the model: a classifier fed
nothing but the DICOM header already reaches `0.557` on the same stratified comparison,
because sicker patients get portable AP films. What the images actually contribute is the
difference.

That margin is also the *more stable* measurement. Its spread across folds is 25% smaller
than the raw stratified AUC's (0.011 vs 0.015), because fold difficulty cancels out. Fold 1
looked like a better model until the pixel-blind baseline rose on that fold too. Its
validation set was simply easier.

---

## The task, and one decision that shapes everything

RSNA labels each film as `Normal`, `Lung Opacity` (infiltrate, with a radiologist-drawn box),
or `No Lung Opacity / Not Normal`, meaning abnormal but not pneumonia. That middle class is
~11,800 of the 26,684 images, and what you do with it decides what your model has learned.

> Here: `Lung Opacity` = 1. `Normal` *and* the middle class = 0.

Clinically the question is "pneumonia, yes or no", not "ill, yes or no". Dropping the middle
class answers the second question while claiming to have answered the first.

The choice is also measurable. Removing the middle class pushes the header-only confounder
from 0.729 to 0.824. The apparently cleaner task is the more biased one, and the reason is in
the cross-tabulation: 48.2% of AP films are middle class. Those are sick patients at the
bedside *without* pneumonia, the counterexamples that stop "AP" from collapsing into
"pneumonia". Delete them and projection alone predicts opacity in 74% of AP films. Choosing
the easier task buys you a shortcut worth 0.095 AUC.

---

## What was found, in order of how uncomfortable it was

### 1. On the first dataset, the file dimensions gave the answer away

The project started on the Kermany paediatric dataset. The model classified well and Grad-CAM
never pointed at anything. The reason:

> The JPEG dimensions alone separate the classes at AUC 0.915. In the official test folder,
> image width alone reaches 0.950, beating the CNN's 0.942.

Not a bug in the code. NORMAL films carry systematically about 2.4× the pixel count of
PNEUMONIA films (median 1654×1323 vs 1160×776, so roughly 1.4× per side). Probably different
acquisition setups, devices or age groups, though the dataset does not say and I could not
verify it. Anything derivable from that size difference (texture sharpness, aspect ratio,
even the shape of a lung mask) is a route to the right answer that has nothing to do with the
chest.

This is why the project moved to RSNA, where every image is 1024×1024 and the confounders are
written in the DICOM header, so they can be measured instead of reconstructed.

### 2. The confounder that remained, and why I stratified instead of hiding it

On RSNA the header-only classifier reaches 0.729, and it is almost entirely projection:

| Header feature | AUC → pneumonia |
| --- | --- |
| `ViewPosition` (AP/PA) | 0.706 |
| age | 0.530 |
| pixel spacing | 0.517 |
| sex | 0.510 |
| all combined | 0.729 |

Within a single projection, the header-only score collapses to ~0.556 on the full set, and to
0.557 averaged over the five validation folds, which is the figure used as the baseline above.
So the confounder is binary and exactly known, which means stratification rather than
matching. Matching on a continuous nuisance variable (as the first dataset required) discarded
two thirds of the data; stratifying on a binary one does not cost a single image.

+0.035 ± 0.006 of the headline number is projection (0.880 unstratified vs 0.845 stratified;
in fold 0 alone it reaches 0.044). That is the difference between the number I could have
published and the number I report.

### 3. The model reads the lung, measurably but diffusely

RSNA provides radiologist-drawn boxes, so the heat map can be scored instead of admired.

| | measured | chance | ratio |
| --- | --- | --- | --- |
| Grad-CAM peak inside a box | 0.539 ± 0.036 | 0.117 ± 0.005 | 4.6× |
| Grad-CAM mass inside a box | 0.192 ± 0.013 | 0.117 ± 0.005 | 1.6× |

The chance baseline is the fraction of image area the boxes cover, and it must be reported. A
hit rate of 0.6 sounds impressive and would be nearly nothing if the boxes covered 55% of the
image.

The honest reading: the peak lands on the pathology in just over half of cases (4.6× chance,
but still only half), and the rest of the map is diffuse. That is a weaker and more accurate
claim than "Grad-CAM looks plausible", which is what a gallery of cherry-picked heat maps
would have supported. Both numbers replicated across five folds.

Related: RSNA films carry burnt-in markers. Ablating the image corners changes the AUC by
-0.0001 ± 0.0009, about as precisely zero as an ablation can come out. No measurable
contribution, which is a weaker statement than "the model ignores them", and the strongest one
the measurement supports.

### 4. Two of my own ideas were wrong, and the disproof is the result

Lung segmentation as preprocessing does not help here. It actively hurts. This was my plan for
a long time, and all three mechanisms by which it could have worked were refuted individually
(saving 11.5 hours of compute that was already scheduled).

Then a better question came up while looking at the alternative: a rectangle around the lungs
necessarily contains the mediastinum. But pixel-exact masking does not *remove* the
mediastinum either. It turns it into a hole, and the shape of that hole is the cardiac and
vascular silhouette. So masking may not delete the confounder at all. It may just re-encode it
as a contour, and a contour is the *easier* feature for a convolutional network to read, not
the harder one.

Measured on 22,846 images, using three clinically readable properties of the cutout
(cardiothoracic ratio, area, vertical position). No grey value enters:

| | AUC | for comparison |
| --- | --- | --- |
| Cutout silhouette → projection | 0.692 | framing of the rectangle: 0.714 |
| Cutout silhouette → pneumonia, stratified | 0.593 | rectangle's crop parameters: 0.551 |

And separately: masking would delete 12.9% of the annotated box area (7.8% into the
mediastinal cutout, 5.1% beyond the lung outline altogether).

So masking keeps about 90% of the above-chance projection signal (0.692 - 0.5 against
0.714 - 0.5), builds a *stronger* new shortcut than cropping does, and deletes an eighth of
the finding. That eighth is concentrated rather than spread out: in 9.2% of positive films
more than a fifth of the box lies in the mediastinal region. Those are the retrocardiac and
paramediastinal consolidations.

The gap between the two shortcut figures, silhouette 0.593 against crop parameters 0.551, is
+0.042, bootstrap 95% CI [0.030, 0.051], so it is not a rounding artefact.

These are three hand-picked summary numbers. A network would see the whole contour, so 0.692
and 0.593 are lower bounds.

A second wrong prediction, for the record. The Kermany aspect ratio differs strongly by class
(mean 1.25 for normal films vs 1.51 for pneumonia) and the pipeline stretches everything to a
square, so I expected the class-correlated distortion to hurt on external validation. Padding
instead of stretching, over 5,856 images: -0.0001. Measured, discarded.

### 5. External validation: the discrimination transfers

Five RSNA-trained checkpoints, pure inference, no fine-tuning, on 5,856 Kermany images (3,054
patient groups). Every axis shifts at once: adults → children aged 1 to 5, USA → Guangzhou,
1024² DICOM → JPEG of varying size, prevalence 0.225 → 0.730.

| | AUC |
| --- | --- |
| Raw ensemble, 5 checkpoints | 0.923 [0.916 to 0.930], bootstrap grouped by patient |
| Single folds | 0.886 ± 0.019, so ensembling is worth more here than internally |
| Metadata leak alone (the 0.915 problem, still present) | 0.914 |
| Leak-adjusted | **0.885** |
| Internal comparison (RSNA, stratified) | 0.845 ± 0.015 |

The ranking transfers. What it does *not* license is a clean "better than internal" claim: the
external task has no middle class (Kermany is normal vs pneumonia only), which this project
argues elsewhere is the easier and more biased framing, and prevalence differs threefold. Read
the external number as "no collapse across a hard domain gap", not as an improvement.

And a check worth more than the headline: the model score and the leak are largely
complementary channels. Model alone 0.923, dimensions alone 0.914, both together 0.966. The
model still adds ~0.05 on top of the leak, and the leak ~0.04 on top of the model. If the
model were simply re-reading the file dimensions, that first increment would not exist.
(Complementary is what is measured; strict independence is not, and additivity alone would not
establish it.)

*One error of mine here, since it changed the number:* the leak-stratified AUC was first
weighted by stratum size. But one stratum held over a thousand positives against a single
negative. Its whole AUC rested on comparisons with that one image, and size-weighting gave
that noise full weight (≈0.85). Weighting instead by discordant pairs, which is the actual
information content of a stratum, gives 0.885. A regression test now reconstructs the case.
The corrected figure is the robust one: it moves by less than 0.002 across every stratum
definition I tried, while the size-weighted variant swings by more than 0.03.

### 6. The calibration does not transfer, and clinically that is the part that matters

Carrying the operating threshold (0.483) across, unchanged:

| | |
| --- | --- |
| Sensitivity | 0.649 |
| Specificity | 0.950 |
| PPV | 0.972 |
| NPV | **0.500** |

> Of the cases this model calls negative, half actually have pneumonia.

The ranking transferred beautifully. The threshold did not: prevalence went from 0.225 to
0.730, and the score distribution shifted with it (sensitivity fell from ~0.82 internally to
0.649, which prevalence alone cannot explain). This is the distinction that usually disappears
between "AUC 0.92" and "ready to use", and it is the reason the demo application in this
repository reports a probability with its uncertainty and never a binary label.

The same problem exists *within* RSNA, between projections. In one fold, a single shared
threshold gave sensitivity 0.894 in AP films and 0.677 in PA, one number behaving as two
different tests. Prevalence differs fourfold between the two strata (0.383 vs 0.093); the
score distributions differ with it, so one shared cut-off sits in a different place in each.

Setting the threshold per projection (on a separate selection split, never on the reporting
set) narrows the sensitivity gap from 0.299 ± 0.073 to 0.067 ± 0.038. Those are means over the
four folds that carry per-projection thresholds; fold 0 predates the change.

---

## Method rules I held to

- Splits are patient-grouped. No patient appears in both training and validation, and I
  checked rather than assumed.
- The holdout is still sealed. 3,812 images, untouched, to be evaluated exactly once at the
  end. Using it for an intermediate check would quietly burn the only unbiased estimate in the
  project.
- Selection and reporting use separate sets. The checkpoint is chosen on an inner
  patient-grouped selection split; the outer validation fold is only reported. Thresholds come
  from the selection split. Optimistic "oracle" variants run alongside so the gap stays
  visible.
- Comparisons are paired within a fold, because fold difficulty varies more than the effects
  being measured. An unpaired difference of 0.005 means nothing.
- Every number sits next to its null hypothesis: header-only baseline, chance box coverage,
  leak-only AUC. Where I could not construct a null, I say so.
- Endpoints are written down before the run. The crop experiment currently running has its
  primary endpoint, its expected effect size, and its "this would refute the idea" case
  committed to the source file, dated before the first result existed.
- Predictions are saved. Every run writes per-image scores, so re-analysis costs seconds
  instead of another nine-hour run. That is what the first re-analysis on the earlier dataset
  actually cost, before this rule existed.

---

## In progress

A rectangular lung crop (the square bounding box of the lung mask plus margin, not a
pixel-exact mask) is being evaluated as a way to close the *external* part of the projection
channel: where the thorax sits in the frame and how large it appears. Five folds, same 8-epoch
budget as the baseline, paired.

Pre-registered, before any result: the primary endpoint is AUC(model score → projection),
currently 0.8166 ± 0.0098, and it should *fall*. A partial drop is the methodologically
expected outcome, because the mediastinum stays inside the crop and the projection cues
*within* the thorax survive untouched. A drop all the way to 0.5 would be suspicious, not
triumphant. No drop refutes the idea, and that would be a result too.

Geometry checks already done, before training: the crop preserves 99.7% of annotated box area,
and its parameters predict the class at 0.552 stratified, so it does not build a new shortcut
of its own.

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
                              perturbation battery, per-epoch history
  rsna_gradcam_grid.py        heat maps for visual inspection, hits and misses separated

Does segmentation help? (answer: no, and it hurts)
  rsna_make_masks.py          U-Net lung masks, resumable, raw output cached
  rsna_cam_lung_check.py      does the heat-map peak fall outside the lung?
  rsna_mask_sweep.py          is the apparent headroom just a mask artefact?
  rsna_mask_silhouette.py     what pixel-exact masking would really do

The rectangular crop
  rsna_crop_geometry.py       can a crop reduce the projection confounder at all?
  rsna_make_crops.py          precomputed crop, bounding boxes rewritten to match
  rsna_crop_compare.py        paired comparison, endpoints fixed in the module header

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

## Hardware, and why it is in the README

A Radeon RX 5500 XT. RDNA1, so no CUDA and no ROCm, which leaves `torch-directml` over
DirectX 12 as the route. No mixed precision, batch size 16, DataLoader workers disabled on
Windows.

It is here because it shaped the design. Every hour of compute had to be justified before it
was spent, which is why this project measures whether an idea can work *before* training on
it. The segmentation question was settled by three cheap measurements instead of an
11.5-hour run. Having to argue for each run made me plan better than I would have otherwise.

---

## Honest limitations

- Single architecture (ResNet-18, ImageNet-initialised). No architecture search, which would
  not have answered any question this project asks.
- The holdout is still sealed, so the final unbiased number does not exist yet.
- External validation is a single dataset, and a paediatric one. It is a hard domain gap, not a
  representative one.
- The Grad-CAM mass ratio of 1.6× is weak. The map localises with its peak and is diffuse
  elsewhere; I have not fixed that.
- Radiologist-drawn boxes are the reference standard, not microbiology or follow-up. The model
  is being measured against *reader opinion*, with all that implies.
- Everything here is one person learning. Assume errors remain; the tests and the saved
  predictions exist so that they can be found.
