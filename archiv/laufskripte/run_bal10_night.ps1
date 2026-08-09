<#
  The alpha = 1.0 series over all five folds, one night, unattended.

  Why this run: alpha = 0.5 was picked on fold 0 with the Grad-CAM damage as
  the counterweight, and over five folds that damage did not replicate
  (t = -1.38). On fold 0 alpha = 1.0 moved the primary endpoint more than
  twice as far (-0.0695 against -0.0405) at a secondary loss of 0.0098, which
  is less than alpha = 0.5 costs on average. One fold does not decide that,
  hence the full series.

  Paired against the existing baseline: same folds, same seed 0, same 8
  epochs, same code. Only the switch changes.

  Usage, from the repository root:
      powershell -ExecutionPolicy Bypass -File .\run_bal10_night.ps1

  Restartable: a fold whose prediction CSV already exists is skipped, so after
  a crash or a reboot the same command picks up where it stopped.

  The red NativeCommandError blocks PowerShell 5.1 produces for native
  stderr are suppressed by converting each line to a string. See the
  comment at the call itself.

  After every fold the script checks that the switch actually took effect
  (stream AUC 0.500, effective sample size around 13,133) and stops the whole
  night if it did not. A run that silently trains the wrong thing costs seven
  hours and looks exactly like a successful one.
#>

$py = ".\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

New-Item -ItemType Directory -Force -Path logs | Out-Null
$start = Get-Date
Write-Host "start $($start.ToString('yyyy-MM-dd HH:mm'))  python: $py"

foreach ($f in 0, 1, 2, 3, 4) {
    $done = "predictions_rsna_bal10\rsna_f${f}_s0.csv"
    if (Test-Path $done) {
        Write-Host "fold ${f}: already done, skipped"
        continue
    }

    $log = "logs\bal10_f${f}.log"
    Write-Host "=== fold ${f}  start $(Get-Date -Format 'HH:mm')  -> $log ==="

    # Two details that are not cosmetic.
    #
    # `| ForEach-Object { $_.ToString() }`: PowerShell 5.1 turns every stderr
    # line of a native program into an ErrorRecord as soon as `2>&1` feeds a
    # pipeline, and prints it as a red NativeCommandError block naming this
    # script. Torch writes its harmless DirectML fallback warning to stderr,
    # so the very first minute of a healthy run looks like a crash. Converting
    # each object to a string before it reaches the console makes it what it
    # is: a line of text.
    #
    # `-u`: without it Python block-buffers stdout whenever the output is a
    # pipe, so the log stays empty for an hour and the numbers that matter
    # (stream AUC, effective sample size) cannot be read while the fold runs.
    & $py -u rsna\pipeline\rsna_train.py --fold $f --epochs 8 --batch 16 --workers 0 `
        --balance-view --balance-strength 1.0 `
        --pred-dir predictions_rsna_bal10 --tag _bal10 2>&1 |
        ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath $log

    if ($LASTEXITCODE -ne 0) {
        Write-Host "fold ${f} FAILED with exit code $LASTEXITCODE. Stopping. See $log"
        break
    }

    # Smoke test on the EFFECT, not on the printed setting.
    $text = Get-Content $log -Raw
    $ok = $true
    if ($text -match 'in the stream:\s*([0-9.]+)') {
        $stream = [double]$Matches[1]
        Write-Host "  check: AUC(ViewPosition -> label) in the stream = $stream (want 0.500)"
        if ($stream -gt 0.55) { $ok = $false }
    } else {
        Write-Host "  check: the balance-view block is missing from the log"
        $ok = $false
    }
    if ($text -match 'effective sample size\s+([0-9]+)') {
        $neff = [int]$Matches[1]
        Write-Host "  check: effective sample size = $neff (want about 13,133, NOT 14,762)"
        if ($neff -gt 14000) { $ok = $false }
    }
    if (-not $ok) {
        Write-Host "ABORT: the switch did not take effect on fold ${f}. Nothing further is started."
        break
    }
}

$have = @(0, 1, 2, 3, 4 | Where-Object { Test-Path "predictions_rsna_bal10\rsna_f${_}_s0.csv" })
Write-Host "`nfolds finished: $($have.Count) of 5   runtime $((Get-Date) - $start)"

if ($have.Count -eq 5) {
    Write-Host "=== paired comparison against the baseline ==="
    & $py -u rsna\befunde\rsna_crop_compare.py --a predictions_rsna --b predictions_rsna_bal10 `
        --name-b "balance-view 1.0" --out predictions_rsna_bal10\compare.csv 2>&1 |
        ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath logs\bal10_compare.log
    Write-Host "written: predictions_rsna_bal10\compare.csv"
} else {
    Write-Host "comparison not run, some folds are missing. Start the script again."
}
