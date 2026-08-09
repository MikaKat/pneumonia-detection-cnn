# Smoke tests

Before every long run, one fold at three epochs. The point was never the number,
it was to find out within twenty minutes whether the switches were wired up, the
paths existed and the output columns were what the analysis would later read.

They are kept because one of them taught a rule that changed how the interim
reports were written.

## The rule they taught

A smoke test at three epochs reports a confounder value that is too high, and
predictably so.

| arm | smoke test said | the five-fold run said |
| --- | --- | --- |
| augmentation | +0.0360 | -0.0052 |
| fixed crop | +0.0836 | +0.0099 |

Both times the smoke test suggested the arm was making things clearly worse.
Both times the finished run said otherwise.

The cause is readable in the run itself. When `best_epoch` sits at 1 of 3, the
checkpoint comes from a model that has barely started to fit, and such a model
leans harder on the easiest available signal, which is the projection. The
overestimate is 0.04 to 0.07.

**So: read `best_epoch` before quoting anything from a smoke test.** If it is 1 of
3, the confounder number is not usable, and the smoke test has only answered the
question it was built for, namely whether the plumbing works.

## Why they are kept rather than deleted

They are the evidence for the sentence above, and they cost nothing to keep. A
rule with two documented cases behind it is worth more than a rule stated in a
memo.

## Contents

| Path | The run it preceded |
| --- | --- |
| `predictions_p5_rauchtest/`, `_ex`, `_em` | the second head, both variants |
| `predictions_p6_rauchtest/` | stronger augmentation |
| `predictions_p7_rauchtest_fix080/` | the fixed-size crop |
| `predictions_rsna_dml1_rauchtest/` | the hardware switch |
| `predictions_cam_smoke/` | the localisation instrument |
| `results_rsna_rauchtest.csv` | the metrics rows of all of them |

These are the only directories in this archive that may be deleted without
losing anything that is referenced elsewhere. The corresponding
`checkpoints\*rauchtest*.pth` can go with them.
