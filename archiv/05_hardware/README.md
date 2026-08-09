# The hardware switch, and the control run that had to follow it

Not a scientific question, but it had to be answered like one: every later
comparison in this project is paired against runs made before the switch, so the
switch itself must not move the numbers.

## The switch

The machine has two DirectML adapters, the integrated graphics and a Radeon RX
5500 XT. Until this point the training had been running on the integrated one,
because `--device directml` picks adapter 0 and nobody had checked which that
was.

The discrete card is 51 percent faster per epoch. Every run since needs
`--dml-index 1`, and a run without it silently uses the slow adapter.

Transfers to the card account for 29 percent of the step time, which is the
ceiling on what any further batch tuning could buy.

## The control run

Speed is uninteresting if the results move with it. A full arm was retrained on
the new adapter and compared against the identical recipe on the old one, paired
per fold, on both endpoints.

Both endpoints came out equivalent. Equivalence was tested with a margin rather
than tested for significance, which is the right test here: the claim is
sameness, and a non-significant difference is not evidence of sameness. That
distinction had already cost this project once.

The control run does **not** count as a comparison partner for the second head.
It shares the recipe but not the seed history, and the phase 5 comparison needed
a partner that differs in the head and in nothing else.

## A prediction that later came true

At the time it was predicted that 512 pixels would cost 3.4 times the step time.
When [08_aufloesung](../08_aufloesung/) actually ran, it measured 2.8 times. The
prediction was in the right range and slightly pessimistic, and a counter-note
written in between claiming otherwise was withdrawn.

## Contents and re-running

| Path | What it is |
| --- | --- |
| `predictions_hardware/` | the timing measurements, per epoch and per step |
| `predictions_rsna_dml1/` | the control arm on the discrete card |

```powershell
venv\Scripts\python.exe rsna\befunde\rsna_hardware.py --dml-index 1
venv\Scripts\python.exe rsna\befunde\rsna_bezugsarm_vergleich.py --apu archiv\00_erste_laeufe_und_nebenanalysen\predictions_rsna_base --karte archiv\05_hardware\predictions_rsna_dml1
```
