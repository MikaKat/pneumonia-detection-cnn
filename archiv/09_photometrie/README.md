# Strong brightness and contrast jitter

The fourth and last attempt on the image, and the only one where the lever was
demonstrated to work before a single training hour was spent. It still failed.

## The idea

Show the network the same film at randomly different brightness and contrast
every epoch, with factors between 0.4 and 1.6 instead of the 0.85 to 1.15 used
until then. If the global tone of an image wobbles anyway, it cannot serve as a
reliable hint about the projection.

## The pre-flight measurement, which is the point of this folder

[06_augmentierung](../06_augmentierung/) failed and then turned out to have
removed only 24 percent of the hint it was aiming at. So this time the reach was
measured first, on all 22,872 development images:

| global quantity | AUC to projection | left at strength 0.15 | left at 0.60 |
| --- | --- | --- | --- |
| mean (brightness) | 0.4604 | 96% | 36% |
| standard deviation (contrast) | 0.2420 | 78% | 26% |

Two things fell out of this before the arm ran.

**The strong global channel is contrast, not brightness.** An AUC of 0.2420
reads as 0.7580 with the sign reversed; 0.50 is the coin flip. The mean is
almost uninformative.

**At the strength used in every run of this project up to that point, the jitter
was effectively not there at all.** 96 percent of the brightness channel
survives it.

That measurement also caught a trap in its own instrument. A first version
perturbed images in a way that clipped at 0 and 255, which quietly changed the
quantity being measured. It was caught because the new instrument was checked
against a case whose answer was already known.

## Written down before the run

Strength 0.60 at 224 pixels. One primary endpoint: the projection channel falls.
Guard on stratified AUC, grey zone at -0.015. And, unusually, a section on what
to expect **if the lever works and the channel does not fall anyway**, so that
the outcome could not be explained away afterwards. That is the case that
occurred.

## Result

| | anchor | this arm | paired difference | |
| --- | --- | --- | --- | --- |
| score to projection | 0.7467 | 0.7413 | -0.0054 [-0.0384, +0.0276] | failed |
| stratified AUC | 0.8368 | 0.8392 | +0.0024 [-0.0012, +0.0060] | guard held |

Three independent pieces of evidence that the lever engaged, all specified
before the run: the pre-flight table above, the strength drawn from the actual
transform in the loader (0.602 and 0.638 measured against the switch value 0.60,
in all five folds, checked in the first half-minute of every run), and the
finished models reacting differently to a fixed brightness shift, where the
sensitivity falls to about a quarter (-0.0109 to -0.0029).

So the channel is not the global grey value. A four times stronger photometric
lever lands on the same point estimate as the weak one did in
[06_augmentierung](../06_augmentierung/), -0.0054 against -0.0052.

## How weak this no is

Honestly: weaker than the one in [08_aufloesung](../08_aufloesung/). The interval
runs from -0.0384 to +0.0276 and excludes almost nothing. A drop of 0.03, more
than half of the largest confounder effect ever measured here, is still
compatible with these data. Whoever tells it more strongly tells it wrong.

## The methodological finding, which changes what a future arm costs

Why is the interval on the classification endpoint ten times narrower than on
the confounder endpoint, when both come from the same five runs?

| | interval half-width |
| --- | --- |
| stratified AUC | 0.0036 |
| score to projection | 0.0330 |

Pairing only works if a fold that runs high in one arm also runs high in the
other. It does on the classification endpoint and it does not on the confounder
endpoint, in all four arms:

| arm | SD of the difference, stratified AUC | SD of the difference, projection |
| --- | --- | --- |
| augmentation | 0.0039 | 0.0245 |
| crop | 0.0058 | 0.0258 |
| 512 pixels | 0.0035 | 0.0164 |
| photometry | 0.0038 | 0.0346 |

In one sentence: the classification endpoint is a property of the fold, the
confounder endpoint is mostly training noise. Which patients land in the
validation set decides the first and is identical in both arms. How strongly a
given run happens to grip the projection hint depends on the random path of that
training run and is independent between arms.

**More folds do not sharpen the confounder endpoint. More seeds do.** Three seeds
per fold would give fifteen paired units and an interval half-width of about
0.016 instead of 0.033, at roughly fourteen hours. That is the entry price for
the next question of this kind, and it was previously filed as a refinement.

## A side benefit that did not change the recipe

The jitter quarters the model's sensitivity to a global brightness shift, which
is a real argument for an application that receives uploads from unknown
equipment. It was not taken into the final recipe anyway, because the winner had
been fixed by a rule written before any of the four arms ran, and taking it now
would have been a selection made after seeing the result.

## Contents and re-running

`predictions_p9_photo/` holds the five folds, `histogramme/` the per-image
intensity histograms behind the pre-flight table.

```powershell
venv\Scripts\python.exe rsna\befunde\rsna_photometrie_reichweite.py
venv\Scripts\python.exe rsna\befunde\rsna_phase9_auswertung.py --pred-dir archiv\09_photometrie\predictions_p9_photo
```
