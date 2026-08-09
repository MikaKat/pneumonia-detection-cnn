# Stronger geometric augmentation

First of four pre-registered attempts to take the projection out of the image.
It failed at its own primary endpoint, and the diagnosis it produced is what
made the fourth attempt worth running.

## The idea

A deterministic transform can re-encode information but not delete it, which had
been measured three times by then. Deleting needs added noise. So: raise the
translation and scale jitter of the training augmentation, so that the apparent
lung size and position wobble from epoch to epoch and stop being a reliable hint
about the projection.

The strengths became command line switches (`--aug-translate`, `--aug-scale`,
`--aug-degrees`) whose defaults are bit-identical to the previous hard-wired
values. That was verified in the test suite rather than assumed, because
otherwise the phase 5 arm would have stopped being a valid comparison partner
and a fresh reference run would have been needed.

## Written down before the run

Anchor: stratified AUC 0.8368, projection 0.7467, the arm from
[03_zweiter_kopf](../03_zweiter_kopf/). Primary endpoint: the projection channel
falls. Guard: stratified AUC must not fall.

The anchor was changed for this phase, and that decision is itself worth
recording. The older gate, "the projection must fall from 0.8166", had saturated:
the reweighting in [04_umgewichtung](../04_umgewichtung/) had already pushed it
to 0.7467, so any arm would have cleared the old bar without doing anything. A
gate that cannot separate decides nothing.

## Result

| | anchor | this arm | paired difference | |
| --- | --- | --- | --- | --- |
| score to projection | 0.7467 | 0.7415 | -0.0052 [-0.0286, +0.0181] | failed |
| stratified AUC | 0.8368 | 0.8409 | +0.0040 | guard held |

Interval half-width 0.0234.

## The diagnosis, which is the useful part

The obvious objection to a null result is that the lever never engaged. Measured
afterwards: the stronger augmentation removes only **24 percent** of the size
hint. So the arm is a weak test of a reasonable idea rather than a refutation of
it.

That is exactly why [09_photometrie](../09_photometrie/) was built the other way
round, with the reach of the lever measured on all 22,872 development images
*before* any training time was spent. This folder is the reason that rule
exists.

## The box trap, avoided on purpose

Raising the geometric jitter is the point in this project where the localisation
head could have been silently destroyed. `RandomAffine` draws its parameters
inside its own call: applying it once to the image and once to the box mask
draws twice and moves the two by different amounts. The box would then sit where
the opacity is not, the head would be supervised against noise, and nothing in
the output would say so. The loss falls, the script finishes, only the map stays
diffuse.

The fix is to draw once with `get_params` and apply the same parameters to both.
The test checks it at the phase 6 strength specifically, and the counter-test
shows the size of what was avoided: two separate draws land 18.74 pixels apart
at this strength, against 7.81 at the previous one.

## Contents and re-running

`predictions_p6_aug/` holds the five folds.

```powershell
venv\Scripts\python.exe rsna\befunde\rsna_phase6_auswertung.py --pred-dir archiv\06_augmentierung\predictions_p6_aug
```
