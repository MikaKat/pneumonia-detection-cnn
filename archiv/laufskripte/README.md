# The scripts that drove the long runs

PowerShell, one per phase, each running five folds unattended overnight. They
are here rather than in the main directory because none of them is needed to
train, serve or reproduce the shipped model. They are how the experiments were
actually executed, which is a different thing.

Every one of them writes into a directory that has since moved into this
archive, so their paths no longer point anywhere. They are kept as a record, not
as something to run.

| Script | What it ran |
| --- | --- |
| `run_phase2.ps1` | the localisation instrument, [02_lokalisation](../02_lokalisation/) |
| `run_phase4.ps1`, `run_bezugsarm_dml1.ps1` | the hardware switch and its control arm, [05_hardware](../05_hardware/) |
| `run_phase5.ps1`, `run_p5_fold1_neu.ps1` | the second head, and the retraining of the fold with the broken weight, [03_zweiter_kopf](../03_zweiter_kopf/) |
| `run_phase6.ps1` | [06_augmentierung](../06_augmentierung/) |
| `run_phase7.ps1` | [07_zuschnitt](../07_zuschnitt/) |
| `run_phase8.ps1` | [08_aufloesung](../08_aufloesung/) |
| `run_phase9.ps1` | [09_photometrie](../09_photometrie/) |
| `run_bal10_night.ps1`, `run_night_queue.ps1` | [04_umgewichtung](../04_umgewichtung/) |
| `run_baseline_redo_night.ps1` | the baseline, [00_erste_laeufe_und_nebenanalysen](../00_erste_laeufe_und_nebenanalysen/) |
| `commit_phase5_6.ps1` | the commit of phases 5 and 6 |
| `patch_header_measured.py` | a one-off migration of the results table |

## What they encode, and why that is the interesting part

They accumulated four defences against failure modes that had already happened
once, and each one is commented in place rather than only in a notebook.

**Resume at fold level.** There is no `--resume` inside a fold. These scripts
check whether a fold's prediction CSV already exists and skip it if so, which is
why an overnight run that died at 3 am cost one fold and not five. The window
between that marker and the metrics row, about a minute wide, was closed in
`run_phase8.ps1`; the earlier scripts still have it.

**No `$ErrorActionPreference = "Stop"`.** In PowerShell 5.1 every line a native
program writes to stderr becomes an error record, and DirectML writes a harmless
fallback warning on the first batch. With `Stop` set, a run dies two minutes in
on a warning that costs nothing. This happened once, after the rule was already
written down but only in a memo, so the reasoning now sits as a comment block at
the exact line where the mistake would be made again.

**Guards formatted with the invariant culture.** PowerShell's format operator
renders `0.080` as `0,080` on a German Windows while Python writes it with a
point. A log guard built the naive way never finds its own line. Found before a
run, not after.

**Helper functions that write nothing to the output stream.** A PowerShell
function returns every line written during its lifetime, not just what follows
`return`. One overnight queue reported "fold 0 failed (0), aborting" after a
successful run, because the return value was the entire log with a zero at the
end.
