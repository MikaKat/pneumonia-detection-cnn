# 512 pixels instead of 224

The third pre-registered attempt on the image. It failed at both of its gates,
and it is the strongest negative result of the four because it excluded its own
hypothesis rather than merely failing to confirm it.

## The idea

At 224 pixels a chest radiograph is heavily downscaled. Fine lung markings
disappear and what survives is largely global tone. If the model reads the
projection off the global grey value, then giving it four times the pixels
should let it read the pathology instead.

## Written down before the run

Two equal gates, either would have counted as a success: stratified AUC rises by
at least 0.008, **or** the projection channel falls. Guard on the other endpoint,
grey zone at -0.015, whole image, no crop.

The bar on the AUC was lowered from 0.01 to 0.008 before the run, because at the
expected precision a bar of 0.01 could not have separated. A gate that cannot
separate decides nothing.

Two effect sizes were named in advance: the AUC was expected to rise by about
0.008 to 0.015, and the projection channel was expected to fall by about 0.033,
based on a 320-pixel probe run earlier.

## Result: both gates failed, and both predictions were excluded

| | anchor | this arm | paired difference | |
| --- | --- | --- | --- | --- |
| stratified AUC | 0.8368 | 0.8336 | -0.0032 | failed |
| score to projection | 0.7467 | 0.7472 | +0.0005 [-0.0151, +0.0161] | failed |

Both pre-registered effects lie **outside** the measured intervals. That is a
stronger statement than a null result: the expected 0.033 drop in the projection
channel is excluded, not just unconfirmed. The 320-pixel probe that suggested it
was a single fold, and this is what a single-fold finding is worth.

## The finding: the confounder moved house

The interesting part is not that nothing changed, but that a lot changed
underneath while the total stayed put.

At 224 pixels the readable channel is the global grey value. At 512 it is fine
texture. Net strength unchanged. Raising the resolution relocates the shortcut
rather than removing it, which closes the resolution axis for this project.

Two side observations that are still unexplained. The Grad-CAM loses its peak at
512 pixels: the lead over the location prior falls from +0.484 to +0.161 while
the mass inside the boxes stays the same. And not one of the 1,500 Grad-CAM maps
degenerates at 512 pixels, against about 2 percent at 224.

## Why this comparison was readable at all

The localisation head pools to a fixed 14 by 14 grid whatever the input size. At
512 pixels `layer3` delivers 32 by 32 and is pooled down. Without that decision,
made two phases earlier, the measuring stick would have moved with the thing it
measures and the arm would have produced an uninterpretable number.

## Cost

1 hour 50 minutes per fold, roughly 9 hours for five, measured at night. A first
estimate taken from a single daytime fold said 3 hours 28 and was wrong by a
factor of two: daytime and nighttime runs on this machine differ that much. A
runtime from one run is not a rule of thumb.

## Contents and re-running

| Path | What it is |
| --- | --- |
| `predictions_p8_s512/` | the five folds at 512 pixels |
| `predictions_rsna_bal10_s320/` | the 320-pixel probe whose suggestion was later excluded |

```powershell
venv\Scripts\python.exe rsna\befunde\rsna_phase8_auswertung.py --pred-dir archiv\08_aufloesung\predictions_p8_s512
```
