# Reweighting the training stream

The only intervention in this project that ever moved the projection confounder.
It is in the shipped model, and it is reported as bought rather than won.

## The idea

Three attempts to remove the projection from the pixels had failed, so this one
addresses the incentive instead of the evidence. The model reads the projection
not because it is visible but because it is useful: in the training stream
`ViewPosition` predicts the label at AUC 0.706, so a model trained on label
accuracy has every reason to encode it.

The imbalance is in the cross-tabulation:

| | label 0 | label 1 | prevalence |
| --- | --- | --- | --- |
| AP | 6,436 | 3,998 | 0.383 |
| PA | 11,282 | 1,156 | 0.093 |

`--balance-view` draws each training image with the weight a chi-square test is
built on, the count expected under independence over the count observed:

    w(v, y) = n_v * n_y / (N * n_vy)

That is 1.26 for AP-negative, 0.59 for AP-positive, 0.85 for PA-negative and
2.42 for PA-positive. Both marginals stay exactly as they were, the overall
prevalence of 0.225 and the AP to PA ratio. Only the association between them is
cut, from 0.706 to exactly 0.500 in the stream. That is deliberate: it leaves
`pos_weight` valid, so the class imbalance is not corrected twice. Weights come
from the fitting split of the current fold alone; selection and reporting splits
are never reweighted.

The price at full dose is 14 percent of the effective sample size by the Kish
measure, 19,698 of 22,872.

## The dose-response curve, fold 0

Raising the weight to the power alpha fixed the operating point before the long
runs:

| alpha | stream AUC | score to projection | stratified AUC | raw AUC | Grad-CAM peak | effective n |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.706 | 0.8076 | 0.8209 | 0.8649 | 0.527 | 15,248 |
| 0.5 | 0.611 | 0.7671 | 0.8194 | 0.8562 | 0.444 | 14,762 |
| 1.0 | 0.500 | 0.7381 | 0.8111 | 0.8434 | 0.340 | 13,133 |

`dosis_wirkung.png` is that curve. The benefit is concave and the harm convex,
so on this fold the midpoint looked efficient: at 46 percent of the dose, 58
percent of the effect arrives against 44 percent of the cost.

## Five folds, both doses, paired within fold

| endpoint | alpha 0.5 | t | alpha 1.0 | t |
| --- | --- | --- | --- | --- |
| score to projection (primary, should fall) | -0.0334 ± 0.0086 | -8.72 | -0.0554 ± 0.0123 | -10.05 |
| stratified AUC (secondary, must not fall) | -0.0144 ± 0.0086 | -3.72 | -0.0181 ± 0.0058 | -7.03 |
| unstratified AUC | -0.0151 ± 0.0061 | -5.54 | -0.0233 ± 0.0021 | -24.66 |

Directly comparing the two doses, again paired within fold: full dose is
reliably better at the primary endpoint (-0.0219 ± 0.0156, t = -3.15) and not
reliably worse at the secondary one (-0.0037 ± 0.0042, t = -1.94).

## The pre-registration was missed, and that is how it is reported

The registered condition was that the stratified AUC must not fall. It fell at
both doses, by 0.0144 at half and by 0.0181 at full. Half dose happened to land
0.0006 inside the 0.015 tolerance and full dose outside it, and that difference
is an arbitrary line rather than the boundary between success and failure. Both
are bought, not won.

Full dose was chosen anyway, and not because it scored better on a composite. It
has a defensible stopping point: complete independence in the stream, `AUC(view
to label) = 0.500` exactly. 0.5 is an arbitrary midpoint. An operating point
that follows from the thing itself is worth more than one that was tuned.

## What it costs beyond the AUC

Measured later on every validation image rather than on 300, the localisation
price is larger than the Grad-CAM peak alone suggested: point AUC in the lung
0.660 against 0.704 for the baseline, hit rate 0.395 against 0.542. It is also
the only variant that fails the control against the lung map on atypically
placed boxes, and the only one that produces heat maps that are zero everywhere,
41 of 5,154 against 0 for the baseline.

So the trade has a number on both sides: a confounder reduced by 0.055 in
exchange for 0.018 of stratified AUC and a measurable loss of localisation
quality. The question of what removing a shortcut costs is usually not asked at
all, which is why this folder exists.

## Contents and re-running

| Path | What it is |
| --- | --- |
| `predictions_rsna_balview/` | the dose-response curve on fold 0 |
| `predictions_rsna_bal05/` | five folds at alpha 0.5 |
| `predictions_rsna_bal10/` | five folds at alpha 1.0, the recipe that ships |
| `dosis_wirkung.png` | the curve as a picture |

```powershell
venv\Scripts\python.exe rsna\pipeline\rsna_train.py --fold 0 --balance-view --balance-strength 1.0 --tag _bal10 --pred-dir archiv\04_umgewichtung\predictions_rsna_bal10
```
