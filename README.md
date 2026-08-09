# Pneumonia detection on chest radiographs

A convolutional network that scores frontal chest radiographs for pneumonic
consolidation and, at the same time, says where. The deployed model is an
ensemble of five calibrated networks. It reaches a stratified AUC of **0.869** on
3,812 held-out images that were sealed on the first day and opened exactly once,
at the end.

The project was built by a freshly graduated doctor entering the clinical work
life, in order to learn how these systems behave rather than to ship a product.

> Not a medical device. Research demonstrator. No diagnostic use.

Contents. [Terms](#terms) · [How the model is built](#how-the-model-is-built)
· [How it was trained](#how-it-was-trained) · [Where the data comes
from](#where-the-data-comes-from) · [Results](#results) · [External
validation](#external-validation) · [What was tried and did not
work](#what-was-tried-and-did-not-work) · [Limitations](#limitations) ·
[Repository](#repository)

---

## Terms

Five terms carry the whole document. Readers who know them can skip ahead.

AUC (area under the ROC curve) is the probability that a randomly chosen patient
with pneumonia gets a higher score than a randomly chosen patient without it. It
is the same quantity as Harrell's c-statistic. 1.0 is perfect and 0.5 is a coin
flip. The 0.5 is the number that matters here, because it is what "knows
nothing" looks like.

A confounder is something that travels with the diagnosis but is not the
diagnosis. On chest radiographs the largest one is projection: AP films are
mostly taken supine with a portable unit, at the bedside, on sicker patients. A
model can learn "this looks like a portable AP film" and score well without ever
looking at the lung.

Stratification is the corresponding correction. Instead of one AUC over
everything, the AUC is computed within AP films and within PA films separately
and then averaged. Inside a projection the confounder cannot help, so what
survives is closer to radiology.

Calibration is a separate property from discrimination, and the two are routinely
confused. Discrimination asks whether the ranking is right; calibration asks
whether a displayed 35% means that 35 in 100 such films really have pneumonia. A
model can rank perfectly and still be wrong about every probability it prints,
and this one was.

Grad-CAM is a heat map showing which image regions drove the score. It is
extracted from a network that was never asked for a location. This project also
asks directly, with a second output trained against the radiologist boxes, and
the difference between the two is one of its findings.

---

## How the model is built

### The trunk: ResNet-18

The backbone is a ResNet-18, initialised from ImageNet weights.

A convolutional network reads an image through small learned filters. Early
layers respond to edges and local texture; each following block halves the
spatial resolution and doubles the number of channels, so later layers see less
of the picture in detail and more of it at once. By the end, a 224 by 224 film
has become a 7 by 7 grid of 512 numbers per position, and those numbers describe
regions rather than pixels.

The "residual" part is what makes 18 layers trainable at all. Each block
computes a correction and adds it to its own input rather than replacing it, so
a block that has nothing useful to contribute can pass its input through
unchanged. Without that shortcut, deep stacks train worse than shallow ones.

The four blocks are named `layer1` to `layer4`. Their output sizes at a 224 by
224 input matter for what follows:

| Stage | Output | Channels |
| --- | --- | --- |
| stem (conv, batchnorm, max pool) | 56 x 56 | 64 |
| `layer1` | 56 x 56 | 64 |
| `layer2` | 28 x 28 | 128 |
| `layer3` | 14 x 14 | 256 |
| `layer4` | 7 x 7 | 512 |

ImageNet initialisation means the filters start out already able to see edges,
texture and shape, learned on photographs. Radiographs are not photographs, but
the early filters transfer well enough that this is worth far more than starting
from noise on a dataset of this size.

### Head one: the diagnosis

`layer4` is averaged over its 7 by 7 positions into a single 512-number vector
and passed through one linear layer to one logit. One and not two: the task
is binary, and a single logit with a sigmoid is the form that lets the class
imbalance be corrected with `pos_weight` in the loss.

### Head two: where

The second output is a **14 by 14 field**, one number per tile, saying how likely
that tile is to contain the opacity. It is a single 1 by 1 convolution on the
256 channels of `layer3`, which is a logistic regression per tile and nothing
more.

Two decisions in that sentence are load-bearing.

It taps `layer3` and not `layer4`. At 224 pixels `layer3` is already 14 by
14, which is the chosen grid, so nothing in the trunk has to be changed to get
it. Reaching 14 from `layer4` would have meant changing that block's stride,
which changes the diagnosis path too, and then the comparison against the same
network without a head would have had two differences instead of one.

It pools to a fixed grid. `adaptive_avg_pool2d` to 14 by 14 means the head
keeps its grid whatever the input size. At 512 pixels `layer3` delivers 32 by 32
and is pooled down. The measuring stick therefore stops moving with the thing it
measures, which is what made the resolution experiment interpretable.

It is the smallest thing that could do the job, deliberately. Anything deeper
would have made "does supervision help" and "does more capacity help" the same
experiment.

```
                    ┌─ layer4 ─ avgpool ─ linear ──────────► 1 logit    (diagnosis)
input ─ stem ─ layer1 ─ layer2 ─ layer3 ─┤
                                          └─ 1x1 conv ──────► 14x14 field (location)
```

Both heads share everything up to `layer3`. That is not an implementation
convenience: the localisation supervision reaches back into the shared trunk and
measurably improves what the diagnosis path looks at, which is reported below.

### What is actually deployed: five of them

The shipped model is an ensemble of the five fold models, each Platt-calibrated
on its own held-out selection split, and the five probabilities are averaged.

Calibrating first and averaging second is not interchangeable with the reverse.
The average of five calibrated probabilities is itself approximately calibrated;
an average of raw scores is on no particular scale. The threshold, 0.2003, comes
from the development data alone and was fixed before the holdout was touched.
Weights, curves and threshold live together in one file,
[`serving/model/kalibrierung_p10.json`](serving/model/kalibrierung_p10.json),
because three numbers that only make sense together should not be three separate
settings.

---

## How it was trained

The whole recipe below is one script,
[`train_final_model.ps1`](train_final_model.ps1), and it is the only training
entry point in the repository. Everything in the results section comes out of
it. Five folds take about three and a half hours on the hardware described at
the end; finished folds are skipped, so it can be interrupted.

### Preprocessing

RSNA ships DICOM. It is converted once to 512 by 512 PNG (`rsna_prepare.py`) as
a faithful downscale: no contrast equalisation, no normalisation, no crop. Every
preprocessing decision belongs in the dataset transform instead, where it can be
switched per run and therefore compared. On the first dataset CLAHE was baked
into the conversion, after which preprocessing and data could no longer be told
apart.

Training then reads those PNGs as greyscale, resizes to 224 by 224 and
normalises with **fixed ImageNet statistics**, not with statistics computed per
image. That looks like the worse choice and was measured to be the better one:
per-image normalisation makes the projection channel stronger, because the
statistic is computed from the image and how much abdomen, shoulder and black
border an image contains depends on the projection.

### Augmentation

Rotation up to 7 degrees, translation up to 3 percent, scale 0.93 to 1.07,
brightness and contrast jitter of 0.15.

**No horizontal flipping.** It produces situs inversus, mirrors the cardiac
silhouette and contradicts the side marker printed into the film.

Image and box mask go through the same affine draw. Calling the transform
twice draws twice and moves the two by different amounts, the box then sits where
the opacity is not, the head is supervised against noise, and nothing in the
output says so: the loss falls, the run finishes, only the map stays diffuse.

### The loss

Binary cross-entropy on the logit with `pos_weight` set to the
negative-to-positive ratio of the fitting split, plus a second binary
cross-entropy on the head field, weighted by a lambda **measured from the first
training batch** rather than chosen, so that the two terms start at comparable
magnitudes. Only films that carry a box contribute to the localisation term.

Optimiser AdamW, learning rate 3e-4, weight decay 1e-4, one-cycle schedule,
8 epochs, batch size 16.

### Decoupling projection from diagnosis in the sampling stream

The model reads the projection not because it is visible but because it is
useful: in the training stream `ViewPosition` predicts the label at AUC 0.706.
Three attempts to remove that information from the pixels had failed, so the
training addresses the incentive instead.

Each training image is drawn with the weight a chi-square test is built on, the
count expected under independence over the count observed:

    w(v, y) = n_v * n_y / (N * n_vy)

Both marginals stay exactly as they were, the overall prevalence of 0.225 and the
AP to PA ratio. Only the association between them is cut, from 0.706 to exactly
0.500. That leaves `pos_weight` valid, so the class imbalance is not corrected
twice. Weights come from the fitting split of the current fold alone.

It works, and it is the only thing in this project that ever moved the
confounder. It is also paid for: see [what did not
work](#what-was-tried-and-did-not-work) for the price and the failed
pre-registration behind it.

### Splitting

`StratifiedGroupKFold` over patient identifiers: five outer folds over the 22,872
development images. Within the training part of each fold, a further
patient-grouped selection split chooses the checkpoint, the calibration curve and
the threshold. The outer fold is reported and never optimised against.

3,812 images were sealed off entirely. The mechanical part of that claim: the
word `holdout` does not appear anywhere in `rsna_train.py`, the only file that
trains or evaluates. The training script cannot touch the holdout because it does
not know it exists.

### The label decision that shapes everything

RSNA labels each film `Normal`, `Lung Opacity` (infiltrate, with a box), or
`No Lung Opacity / Not Normal`, meaning abnormal but not pneumonia. That middle
class is about 11,800 of 26,684 images.

> Here: `Lung Opacity` = 1. `Normal` and the middle class = 0.

Clinically the question is "pneumonia, yes or no", not "ill, yes or no". Dropping
the middle class answers the second while claiming to have answered the first.

It is also measurable. Removing the middle class pushes the header-only
confounder from 0.729 to 0.824, because 48.2 percent of AP films are middle
class: sick patients at the bedside without pneumonia, the counterexamples that
stop "AP" from collapsing into "pneumonia". Choosing the easier task buys a
shortcut worth 0.095 AUC.

---

## Where the data comes from

Nothing here ships image data. All three sets are public and have to be
downloaded from the source, which is also where their licences and terms of use
are stated.

| Set | Used for | Source |
| --- | --- | --- |
| RSNA Pneumonia Detection Challenge | the classifier: 26,684 adult chest radiographs with radiologist-drawn boxes | [kaggle.com/competitions/rsna-pneumonia-detection-challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) |
| Kermany paediatric chest X-ray | external validation only, and the first dataset the project ran on | [data.mendeley.com/datasets/rscbjbr9sj](https://data.mendeley.com/datasets/rscbjbr9sj/2) |
| Montgomery County and Shenzhen | the lung segmenter (U-Net), which the web app shows but which scores nothing | [data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/) |

Papers behind them, in the same order: Shih et al., *Augmenting the National
Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible
Pneumonia*, Radiology: Artificial Intelligence 2019. Kermany et al., *Identifying
Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell
2018. Jaeger et al., *Two public chest X-ray datasets for computer-aided
screening of pulmonary diseases*, Quantitative Imaging in Medicine and Surgery
2014.

The Kermany set appears twice on purpose. It was the project's starting point, it
was abandoned once its file dimensions turned out to give the answer away, and it
comes back only as a held-out external test set that no training run ever saw.
Details in [`archiv/kermany`](archiv/kermany/).

| | RSNA | |
| --- | --- | --- |
| total labelled images | 26,684 | |
| development, five patient-grouped folds | 22,872 | prevalence 0.225 |
| holdout, opened once | 3,812 | 1,739 AP, 2,073 PA |

---

## Results

Everything below was measured once, on the sealed set, after the model, the
curves and the threshold had been written to disk. The full run is in
[`predictions_holdout/`](predictions_holdout/).

### The primary endpoint

| | measured | registered in advance |
| --- | --- | --- |
| **stratified AUC, ensemble** | **0.8687** [0.8566, 0.8805] | lower interval bound above 0.80 |
| stratified AUC, single models | 0.8473 | 0.8368 on average |
| score to projection | 0.7501 [0.7391, 0.7610] | around 0.75 |
| calibration error after Platt | 0.0260 | below 0.03 good, above 0.05 a finding |

Intervals are 90 percent, from a stratified bootstrap over images with 2,000
draws. The bar of 0.80 was three fold standard deviations below the
cross-validated estimate, and it was checked in advance that both outcomes were
reachable: at the expected precision the lower bound would have landed near 0.83
if the model held and below 0.80 if it did not.

### The number that matters more than the headline

Each of the five models against its own cross-validation figure:

| fold | cross-validation | holdout | difference |
| --- | --- | --- | --- |
| 0 | 0.8211 | 0.8557 | +0.0346 |
| 1 | 0.8426 | 0.8439 | +0.0013 |
| 2 | 0.8479 | 0.8561 | +0.0082 |
| 3 | 0.8407 | 0.8388 | -0.0018 |
| 4 | 0.8316 | 0.8421 | +0.0104 |
| **mean** | **0.8368** | **0.8473** | **+0.0105** |

The cross-validated estimate was not optimistic. Nine phases of decisions were
taken against a number that comes back on data none of those decisions could
reach. No fold collapses; the worst sits 0.002 below its own estimate.

What this does not establish: the holdout comes from the same collection, the
same devices and the same labelling procedure. It answers whether the development
process fitted itself to its own data, and the answer is no. It says nothing
about a different hospital.

### Against the null hypothesis

A classifier fed nothing but the DICOM header, no pixels at all, reaches 0.557 on
the same stratified comparison, because sicker patients get portable AP films.

| Header feature | AUC to pneumonia |
| --- | --- |
| `ViewPosition` (AP/PA) | 0.706 |
| age | 0.530 |
| pixel spacing | 0.517 |
| sex | 0.510 |
| all combined | 0.729 (0.557 stratified) |

The margin over that baseline is the informative headline, and it is also the
more stable measurement: its spread across folds is 25 percent smaller than the
raw stratified AUC's, because fold difficulty cancels out.

### Calibration

| on the holdout | raw | calibrated |
| --- | --- | --- |
| mean prediction (observed 0.2251) | 0.3279 | 0.2215 |
| Brier score | 0.1272 | 0.1107 |
| expected calibration error | 0.1029 | 0.0260 |

The raw model holds patients to be considerably sicker than they are. That is not
a bug but the side effect of weighting the rare class up so it is learned at all.
Two parameters fitted on data the model never trained on remove it, and they
still work on data the curve itself never saw.

### Localisation

All 5,154 positive validation images, five folds. The measure is the area under
the curve over pixels inside a lung mask: draw one pixel inside a box and one
outside, how often is the map higher inside. Chance is exactly 0.5.

| | point AUC in the lung | hit rate |
| --- | --- | --- |
| **supervised head** | **0.9123** | |
| location prior | 0.7520 | 0.5714 |
| Grad-CAM, with the head | 0.7312 | 0.5982 |
| lung map | 0.7011 | 0.5326 |
| Grad-CAM, without the head | 0.6786 | 0.4212 |
| chance | 0.5000 | 0.117 |

The opponent is not chance, it is anatomy. The location prior is every training
box drawn into one grid and averaged: a single fixed map, identical for every
image, that knows nothing except where opacities usually sit. Grad-CAM alone sits
**below** it. The supervised head is the first map in this project that clearly
beats it, and it stays ahead in every quintile of how typical the box position is
(0.8753 to 0.9355, while the prior runs 0.5124 to 0.9208).

Adding the head costs nothing on the diagnosis: +0.0081 [+0.0014, +0.0149]
stratified AUC, non-inferiority at a margin fixed beforehand. The honest sentence
is "costs nothing", not "helps".

The head is **not calibrated**, and this is why the application draws it as a
gradient with no box and no cut-off: on films without pneumonia it lights up
somewhere in 62 percent of cases. Scored as a detection task it reaches 0.136
against 0.025 for the location prior, and almost half the loss is false alarms.
Full numbers in [`archiv/03_zweiter_kopf`](archiv/03_zweiter_kopf/).

### The threshold in practice, and why no label is shown

At the pre-registered threshold of 0.2003:

| | sensitivity | specificity | n | prevalence |
| --- | --- | --- | --- | --- |
| together | 0.8648 | 0.7282 | 3,812 | 0.225 |
| AP films | 0.8814 | 0.5825 | 1,739 | 0.383 |
| PA films | 0.8073 | 0.8113 | 2,073 | 0.093 |

A single threshold at unequal prevalence is practically two different tests.
Setting it per projection closes most of the gap and **cannot be deployed**: the
application receives an uploaded PNG with no DICOM header and never learns the
projection. That is the honest reason it reports a probability with its spread
and never a yes or no.

---

## External validation

Five RSNA-trained checkpoints, pure inference, no fine-tuning, on 5,856 Kermany
images from 3,054 patient groups. Every axis shifts at once: adults to children
aged 1 to 5, USA to Guangzhou, 1024² DICOM to JPEG of varying size, prevalence
0.225 to 0.730.

| | AUC |
| --- | --- |
| raw ensemble | 0.923 [0.916, 0.930], bootstrap grouped by patient |
| single folds | 0.886 ± 0.019 |
| metadata leak alone | 0.914 |
| **leak-adjusted** | **0.885** |
| internal comparison (RSNA, stratified) | 0.845 ± 0.015 |

The ranking transfers. What that does not license is a clean "better than
internal" claim: the external task has no middle class, which this project argues
is the easier and more biased framing, and prevalence differs threefold. It reads
as "no collapse across a hard domain gap", not as an improvement.

The leak is real and had to be corrected for. In the Kermany set the JPEG
dimensions alone separate the classes at AUC 0.915; in the official test folder
image width alone reaches 0.950, beating the CNN's own 0.942. Normal films carry
about 2.4 times the pixel count of pneumonia films. That is why the project moved
to RSNA, where every image is 1024 by 1024 and the confounders are written in the
DICOM header, so they can be measured instead of reconstructed.

One check is worth more than the headline: model score and leak are largely
complementary channels. Model alone 0.923, dimensions alone 0.914, both
together 0.966. If the model were simply re-reading the file dimensions, that
increment would not exist. Complementary is what is measured; strict independence
is not.

### And the part that says do not deploy this

Carrying the operating threshold across unchanged:

| | |
| --- | --- |
| sensitivity | 0.649 |
| specificity | 0.950 |
| PPV | 0.972 |
| **NPV** | **0.500** |

> Of the cases this model calls negative, half actually have pneumonia.

The ranking transferred; the calibration did not. This is the distinction that
usually disappears between "AUC 0.92" and "ready to use".

Per-image output: [`archiv/00_erste_laeufe_und_nebenanalysen`](archiv/00_erste_laeufe_und_nebenanalysen/).

---

## What was tried and did not work

Nine interventions were run under pre-registration: a primary endpoint, a bar, a
guard on the endpoint that must not move, and a grey zone saying what would make
the measurement too imprecise to judge, each written in a dated document before
the run. **Four failed at their own bar.** They are reported as failures, with
their raw predictions kept, and each has its own folder.

| What was tried | Result | Details |
| --- | --- | --- |
| Pixel-exact lung masking | Refuted before training. The mediastinum becomes a hole whose shape is the cardiac silhouette, keeping ~90% of the projection signal and deleting 12.9% of the annotated boxes. | [00](archiv/00_erste_laeufe_und_nebenanalysen/) |
| Per-image intensity normalisation | Refuted. Makes the channel stronger, 0.721 to 0.768, and CLAHE worse still at 0.818. | [00](archiv/00_erste_laeufe_und_nebenanalysen/) |
| Adaptive lung crop | Refuted. The confounder rose in all five folds, because window size is itself a projection proxy. | [07](archiv/07_zuschnitt/) |
| Stronger geometric augmentation | Failed. -0.0052 [-0.0286, +0.0181]. Diagnosis: only 24% of the size hint removed. | [06](archiv/06_augmentierung/) |
| Fixed-size crop | Failed. +0.0099 [-0.0147, +0.0345], the point estimate moved the wrong way. | [07](archiv/07_zuschnitt/) |
| 512 pixels instead of 224 | Failed at both bars, and both registered effects fell **outside** the intervals. The confounder moved house: grey value at 224, fine texture at 512. | [08](archiv/08_aufloesung/) |
| Strong photometric jitter | Failed. -0.0054 [-0.0384, +0.0276], with the lever demonstrated to work beforehand. | [09](archiv/09_photometrie/) |

Four interventions at the image, four null results at the confounder. The one
thing that did move it addresses the training stream rather than the pixels, and
it is in the shipped model at a stated price:

| | change against baseline | t |
| --- | --- | --- |
| score to projection (primary, should fall) | **-0.0554 ± 0.0123** | -10.05 |
| stratified AUC (secondary, must not fall) | -0.0181 ± 0.0058 | -7.03 |

Ten times the size of anything the image interventions produced, and **a failed
pre-registration**: the stratified AUC was not allowed to fall and it fell, by
more than the tolerance set in advance. It also costs localisation quality (point
AUC 0.660 against 0.704). The honest summary is bought, not won, and the trade
has a number on both sides. Details in
[`archiv/04_umgewichtung`](archiv/04_umgewichtung/).

One methodological result changes what a further attempt would cost. Pairing
shrinks the interval on the diagnosis endpoint to a third in all four arms and
never on the confounder endpoint, because the latter is training noise rather
than a property of the fold. **More folds do not sharpen it, more seeds do:**
three seeds per fold, about fourteen hours, would roughly halve the interval.
That is the entry price for the next question of this kind, and it means an
effect of 0.03 could have been missed in three of the four arms.

The full index of experiments, with per-image predictions for each, is in
[`archiv/README.md`](archiv/README.md).

---

## Method rules

- Splits are patient-grouped, and this was checked rather than assumed.
- The holdout was opened once, after the model, the curves and the threshold were
  fixed and written to a file. The prediction script locks itself afterwards,
  because the usual way a holdout is spent is not dishonesty but convenience.
- Selection and reporting use separate sets. The checkpoint, the calibration curve
  and the threshold come from an inner patient-grouped selection split; the outer
  fold is only reported.
- Comparisons are paired within a fold, because fold difficulty varies more than
  the effects being measured.
- Every number sits next to its null hypothesis: header-only baseline, location
  prior and lung map for the heat maps, leak-only AUC. One of those nulls, the
  location prior, overturned this project's original headline.
- Endpoints are written down before the run, and a bar is checked for whether it
  can separate before the run. A bar the expected outcome clears no matter what
  decides nothing; one of them had to be lowered in advance for that reason.
- Predictions are saved. Every run writes per-image scores, so re-analysis costs
  seconds instead of another nine-hour run.

---

## Repository

```
train_final_model.ps1   trains the deployed model, five folds, one recipe.
                        Start here to reproduce anything below.
commit.ps1              versions the repository in reviewable steps

rsna/pipeline/       the model's own path
  rsna_prepare.py      DICOM to PNG, once
  rsna_splits.py       patient-grouped splits, holdout sealed off
  rsna_data.py         DICOM header extraction
  rsna_train.py        the two-headed network, training, stratified metrics,
                       Grad-CAM against boxes, perturbation battery
  rsna_holdout.py      the one pass over the sealed set, locks itself afterwards
  rsna_make_crops.py, rsna_make_masks.py, rsna_crop_masks.py

rsna/befunde/        the analyses, including those for the archived experiments
  rsna_metadata_leak_check.py  the header-only baseline, run before any training
  rsna_lokalisation.py         location prior and lung map: the null for the heat maps
  rsna_cam_power.py            every map against every null, on every positive image
  rsna_platt.py                the five calibration curves and the threshold
  rsna_phase10_auswertung.py   the verdict, pre-registration wired in as constants
  rsna_external_kermany.py     leak stratification, grouped bootstrap, threshold transfer
  ...                          one per archived experiment

serving/             FastAPI backend, serves the five-model ensemble
  model/               the network definition and kalibrierung_p10.json, which
                       names the five weights, the five curves and the threshold
  segmentation/        the U-Net lung finder. It sits here rather than at the
                       root because the application is its only consumer: it is
                       shown on its own card and never touches the score
webapp/              the interface
checkpoints/         weights; only the five of the ensemble and the U-Net are versioned
tests/               run against hand-computed cases; each states which silent
                     failure it prevents
archiv/              every experiment that did not end up in the model
predictions_final_model/   the five folds the shipped model is built from
predictions_holdout/       the one pass over the sealed set
results_rsna.csv           one row per training run, every configuration column
qc/rsna/                   the DICOM headers, which are both the input to the
                           split and the evidence for the header-only baseline
```

Every script's module docstring says what it produces, why it exists, and how to
read its result, including which value is the null and what would falsify the
claim.

```powershell
# reproduce the headline, in this order
venv\Scripts\python.exe rsna\befunde\rsna_metadata_leak_check.py  # the null hypothesis, first
venv\Scripts\python.exe rsna\pipeline\rsna_prepare.py             # DICOM to PNG
venv\Scripts\python.exe rsna\pipeline\rsna_splits.py              # folds, and the seal

.\train_final_model.ps1 -Rauchtest    # one fold, three epochs, half an hour
.\train_final_model.ps1               # five folds, about three and a half hours

venv\Scripts\python.exe rsna\befunde\rsna_platt.py           # curves and threshold,
                                                             # development data only
venv\Scripts\python.exe tests\test_rsna_phase10.py           # the guard, must be green
venv\Scripts\python.exe rsna\pipeline\rsna_holdout.py --dml-index 1   # once, and only once
venv\Scripts\python.exe rsna\befunde\rsna_phase10_auswertung.py
```

The order is binding rather than conventional. Curves and threshold have to be
written to disk before the holdout is computed; doing it the other way round
means the holdout picked the curve, and the number stops meaning what it says.
`rsna_holdout.py` locks itself after the first pass, so in this repository it
will refuse to run.

### Hardware, and why it appears in a README

A Radeon RX 5500 XT. RDNA1, so no CUDA and no ROCm, which leaves `torch-directml`
over DirectX 12. No mixed precision, batch size 16, DataLoader workers disabled
on Windows.

It appears here because it shaped the design. Every hour of compute had to be
justified before it was spent, which is why this project measures whether an idea
can work before training on it. The segmentation question was settled by three
cheap measurements instead of an 11.5-hour run, and a pre-set grey zone released
a fourteen-hour follow-up run three times without spending it.

---

## Limitations

- Single architecture. No architecture search, which would not have answered any
  question this project asks.
- The holdout is spent. It was opened once, which is the correct number, and any
  further change to the model has no unbiased set left to be measured on.
- The ensemble has no clean calibration set. Each member is calibrated on data it
  never saw, and the average of calibrated probabilities is then approximately
  calibrated, but that last step was verified on the holdout rather than
  guaranteed by construction.
- The projection confounder is shipped. The score predicts AP against PA at
  0.750, four pre-registered image interventions did not lower it, and the one
  that did is a measured trade rather than a fix. A station with a different AP
  to PA mix should expect a different operating point than the one measured here.
- The single threshold behaves as two different tests, and per-projection
  thresholds cannot be deployed because the application never sees the
  projection.
- External validation is a single dataset, and a paediatric one. It is a hard
  domain gap, not a representative one, and no external set was evaluated for the
  final ensemble at all.
- The localisation head is uncalibrated and fires on 62 percent of films without
  pneumonia, so it is shown as a hint about the region and never as a finding.
- The confounder endpoint was measured too coarsely to exclude small effects.
- The localisation findings rest on RSNA boxes only. A box is a reader's rectangle
  around a region, not a segmentation of the pathology, and the measure treats
  every pixel in it as equally diseased.
- Radiologist-drawn boxes are the reference standard, not microbiology or
  follow-up. The model is measured against reader opinion, with all that implies.
- Everything here is the work of one person learning. Errors should be assumed to
  remain; the tests and the saved predictions exist so that they can be found.
