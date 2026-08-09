# Archive: everything that was tried

The main directory holds the shipped model and nothing else. Every experiment
that did not end up in it lives here, with its raw per-image predictions intact,
so that any number in the main README can be recomputed rather than believed.

Each folder has its own README with what was asked, what was written down before
the run, what came out, and which script produced it. The verdicts below are the
one-line versions.

| Folder | Question | Verdict |
| --- | --- | --- |
| [00_erste_laeufe_und_nebenanalysen](00_erste_laeufe_und_nebenanalysen/) | baseline five folds, and the three preprocessing routes | baseline stands, all three routes refuted |
| [01_klassen_und_kalibrierung](01_klassen_und_kalibrierung/) | where do the errors sit, and is the score a probability? | errors sit in the middle class, the score is not a probability |
| [02_lokalisation](02_lokalisation/) | does Grad-CAM point at the pathology? | it points at this image, and it is weaker than anatomy |
| [03_zweiter_kopf](03_zweiter_kopf/) | does asking the model where beat reading it out afterwards? | yes, clearly, and it costs nothing on the diagnosis |
| [04_umgewichtung](04_umgewichtung/) | can the projection be removed from the training stream? | the only thing that ever moved it, at a measured price |
| [05_hardware](05_hardware/) | is the discrete card worth the switch? | yes, 51 percent faster, same results |
| [06_augmentierung](06_augmentierung/) | does stronger geometric augmentation remove the projection? | no |
| [07_zuschnitt](07_zuschnitt/) | does cropping to the lungs remove it? | no, and the adaptive version made it worse |
| [08_aufloesung](08_aufloesung/) | does 512 pixels remove it? | no, it relocates it |
| [09_photometrie](09_photometrie/) | does strong brightness and contrast jitter remove it? | no, and this time the lever was demonstrated first |
| [10_rauchtests](10_rauchtests/) | the one-fold, three-epoch probes run before each long run | kept because one of them taught a rule |
| [kermany](kermany/) | the first dataset the project ran on | abandoned, the file dimensions gave the answer away |
| [laufskripte](laufskripte/) | the PowerShell scripts that drove the long runs | history, not part of the model |

## How to read a folder

Every `predictions_*` directory has the same shape:

| File | Contents |
| --- | --- |
| `rsna_f{k}_s0.csv` | one row per image of the reporting fold: id, label, projection, score |
| `sel_f{k}_s0.csv` | the same for the inner selection split, which is where thresholds and calibration curves come from |
| `history_f{k}_s0.csv` | one row per epoch |
| `cam_f{k}_s0.csv` | the Grad-CAM table against the annotated boxes |

Because every score reached disk, a follow-up question costs a re-analysis of
seconds instead of a training run of hours. That rule was written after the
first re-analysis on the earlier dataset cost nine hours.

## Re-running an archived analysis

The analysis scripts stayed in `rsna/befunde/`. They were written when these
directories sat in the main directory, so their default paths no longer point
anywhere. Every one of them takes the path as an argument:

```powershell
venv\Scripts\python.exe rsna\befunde\rsna_phase8_auswertung.py --pred-dir archiv\08_aufloesung\predictions_p8_s512
```

The exact call for each experiment is in that experiment's README.

## What is deliberately not here

Weights. `checkpoints/` still holds all 74 of them, roughly 45 MB each, and
`.gitignore` lets exactly the five weights of the shipped ensemble plus the lung
segmenter through. The rest are reproducible from the splits, the seed and the
scripts, which the result CSVs beside them are not. That is the whole reason the
numbers are versioned and the weights are not.
