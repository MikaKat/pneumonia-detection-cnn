<#
  Rebuild the baseline weights that were overwritten before the --tag switch
  existed.

  What happened: every run wrote to checkpoints\rsna_f{fold}_s{seed}.pth, so
  the crop runs of 28.07. and the balance-view run of 29.07. replaced the
  baseline files one by one. The file names never changed, so nothing looked
  wrong. The reported baseline NUMBERS are untouched, they live in
  predictions_rsna\*.csv from 26.07. Only the weights are gone, and Grad-CAM
  needs the weights.

  This run reproduces them. Same seed, same folds, same 8 epochs, no switches.
  The training is deterministic, so the new predictions must match the stored
  ones. rsna_cam_power.py checks exactly that before it measures anything.

  Nothing is overwritten this time: --tag _base and a separate --pred-dir.

  Usage, from the repository root:
      powershell -ExecutionPolicy Bypass -File .\run_baseline_redo_night.ps1

  Restartable: a fold whose prediction CSV already exists is skipped.
#>

$py = ".\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

New-Item -ItemType Directory -Force -Path logs | Out-Null
$start = Get-Date
Write-Host "start $($start.ToString('yyyy-MM-dd HH:mm'))  python: $py"

foreach ($f in 0, 1, 2, 3, 4) {
    $done = "predictions_rsna_base\rsna_f${f}_s0.csv"
    if (Test-Path $done) { Write-Host "fold ${f}: already done, skipped"; continue }

    $log = "logs\base_f${f}.log"
    Write-Host "=== fold ${f}  start $(Get-Date -Format 'HH:mm')  -> $log ==="

    # ForEach-Object turns stderr lines into plain text: PowerShell 5.1 would
    # otherwise print the harmless DirectML warning as a red error block.
    # -u keeps the log readable while the fold runs.
    & $py -u rsna\pipeline\rsna_train.py --fold $f --epochs 8 --batch 16 --workers 0 `
        --pred-dir predictions_rsna_base --tag _base 2>&1 |
        ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath $log

    if ($LASTEXITCODE -ne 0) {
        Write-Host "fold ${f} FAILED with exit code $LASTEXITCODE. Stopping. See $log"
        break
    }

    # Smoke test on the EFFECT: no reweighting may have happened here.
    $text = Get-Content $log -Raw
    if ($text -match 'balance-view at strength') {
        Write-Host "ABORT: this run reweighted, which the baseline must not do."
        break
    }
    Write-Host "  check: no reweighting block in the log, as it should be"
}

$have = @(0, 1, 2, 3, 4 | Where-Object { Test-Path "predictions_rsna_base\rsna_f${_}_s0.csv" })
Write-Host "`nfolds finished: $($have.Count) of 5   runtime $((Get-Date) - $start)"

if ($have.Count -eq 5) {
    Write-Host "=== do the new predictions match the reported ones? ==="
    & $py -u -c @"
import pandas as pd, numpy as np
for f in range(5):
    a = pd.read_csv(f'predictions_rsna/rsna_f{f}_s0.csv').set_index('patientId')['p_clean']
    b = pd.read_csv(f'predictions_rsna_base/rsna_f{f}_s0.csv').set_index('patientId')['p_clean']
    common = a.index.intersection(b.index)
    d = float(np.abs(a.loc[common].to_numpy() - b.loc[common].to_numpy()).max())
    print(f'  fold {f}: n {len(common)}, largest difference {d:.2e}',
          '  IDENTICAL' if d < 1e-6 else '  DIFFERENT, do not use')
"@
    Write-Host "`nnext:  $py rsna\befunde\rsna_cam_power.py"
} else {
    Write-Host "comparison not run, some folds are missing. Start the script again."
}
