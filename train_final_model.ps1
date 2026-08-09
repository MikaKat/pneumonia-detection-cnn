<#
  Trains the model that is deployed. Five folds, one recipe, nothing else.

  This is the script behind everything in README.md. Running it from scratch
  reproduces the five weights that `serving/` loads, the predictions that the
  calibration curves are fitted on, and the metrics rows in results_rsna.csv.

  THE RECIPE
  ----------
    ResNet-18, ImageNet-initialised, two heads
      head 1   one logit, binary cross-entropy with pos_weight
      head 2   14x14 field on layer3, supervised against the radiologist boxes,
               with a lambda measured from the first batch rather than chosen
    --head-negatives exclude    only films that carry a box feed the box loss
    --balance-view              projection and diagnosis are decoupled in the
                                sampling stream, at full strength
    224 pixels, batch 16, AdamW, one-cycle, 8 epochs

  Every one of those five decisions was measured against its alternative before
  it was adopted. The experiments that lost are in archiv/, one folder each.

  WHY THE TAG STILL SAYS p5head_ex
  --------------------------------
  The weights are called rsna_f{0..4}_s0_p5head_ex.pth and the tag written into
  results_rsna.csv is _p5head_ex. That name is not decoration, it is the key
  that ties four things together: the metrics row, the weight file, the
  calibration curve in serving/model/kalibrierung_p10.json, and the checksums in
  the holdout lock file. Renaming it would make the lock file point at files
  that no longer exist, and this project has already lost weights once to a
  careless naming change. The script has a readable name; the provenance key
  keeps its own.

  THE SMOKE TEST FIRST
  --------------------
      powershell -ExecutionPolicy Bypass -File .\train_final_model.ps1 -Rauchtest

  One fold, three epochs, no Grad-CAM, about half an hour. It then checks that
  the head field beats the location prior. If that fails, either the box trap
  has closed (image and mask drawn with different affine parameters) or the head
  says zero everywhere, and the ten hours below would have been wasted.

  DO NOT read the confounder number out of a smoke test. At three epochs the
  checkpoint comes from a model that has barely started to fit, and such a model
  leans harder on the easiest signal. Measured twice, it overestimates by 0.04
  to 0.07: archiv/10_rauchtests.

  THEN THE REAL RUN
  -----------------
      powershell -ExecutionPolicy Bypass -File .\train_final_model.ps1

  About 3 h 16 to 3 h 45 on a Radeon RX 5500 XT. Finished folds are skipped, so
  the run may be interrupted and restarted at any time. There is no resume
  inside a fold: an abort mid-fold costs that fold.

  Single folds, if something has to be repeated:
      powershell -ExecutionPolicy Bypass -File .\train_final_model.ps1 -Folds 2,3
#>

param(
    [int[]]$Folds = @(0, 1, 2, 3, 4),
    [int]$DmlIndex = 1,
    [switch]$Rauchtest
)

# NO $ErrorActionPreference = "Stop" HERE, and that is deliberate.
#
# In PowerShell 5.1 every line a native program writes to stderr inside a
# pipeline becomes an error record. torch-directml writes a harmless fallback
# warning on the first batch (aten::log_sigmoid_forward will run on the CPU).
# With Stop set, the run dies two minutes in on a warning that costs nothing.
# That happened once, on 04.08.2026, after the rule had already been written
# down in a note. A rule that lives only in a note does not always work, so the
# reasoning sits here, at the exact line where the mistake would be made again.
#
# It is not needed either: $LASTEXITCODE is checked after every call and every
# condition of our own ends in an explicit `exit 1`.

$py = ".\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
New-Item -ItemType Directory -Force -Path logs | Out-Null

$PredDir  = "predictions_final_model"
$Tag      = "_p5head_ex"
$Ergebnis = "results_rsna.csv"

if ($Rauchtest) {
    $Folds   = @(0)
    $Epochen = 3
    $CamN    = 0
    $PredDir = "archiv\10_rauchtests\predictions_rauchtest_final"
    $Tag     = "_rauchtest_final"
    Write-Host "RAUCHTEST: 1 Fold, 3 Epochen, kein Grad-CAM, eigener Ordner."
    Write-Host "Die Confounder-Zahl aus diesem Lauf ist NICHT verwertbar."
} else {
    $Epochen = 8
    $CamN    = 300
}

Write-Host ""
Write-Host "Ziel  : $PredDir\   (Tag $Tag)"
Write-Host "Folds : $($Folds -join ', ')   Epochen: $Epochen   Adapter: $DmlIndex"
Write-Host ""

foreach ($f in $Folds) {

    # Fold-level resume. The marker is the prediction CSV, not the checkpoint:
    # the checkpoint is written before the predictions, so a run killed between
    # the two would otherwise look finished. The metrics row is checked as well,
    # because there is about a minute between the CSV and the row.
    $marke = Join-Path $PredDir "rsna_f${f}_s0.csv"
    if (Test-Path $marke) {
        $zeile = $false
        if (Test-Path $Ergebnis) {
            $zeile = @(Import-Csv $Ergebnis |
                       Where-Object { $_.fold -eq "$f" -and $_.tag -eq $Tag }).Count -gt 0
        }
        if ($zeile) { Write-Host "fold ${f}: fertig, uebersprungen"; continue }
        Write-Host "fold ${f}: Vorhersagen da, aber keine Ergebniszeile. Wird neu gerechnet."
    }

    $log = "logs\final_f${f}.log"
    Write-Host "=== fold ${f}  start $(Get-Date -Format 'HH:mm')  -> $log ==="

    # Built as ONE array and splatted. An empty element in the middle of a call
    # line can reach argparse as an empty string in PowerShell 5.1, and it then
    # aborts with a message that looks like a code error and is not one.
    $cmd = @("-u", "rsna\pipeline\rsna_train.py",
             "--fold", $f, "--epochs", $Epochen, "--batch", 16,
             "--workers", 0, "--cam-n", $CamN,
             "--dml-index", $DmlIndex,
             "--balance-view",
             "--head", "--head-negatives", "exclude",
             "--pred-dir", $PredDir, "--tag", $Tag, "--out", $Ergebnis)

    # The call itself belongs IN the log, not only on screen. A provenance that
    # exists only in a command line is a provenance that gets lost. Whoever
    # reads this log months later sees in line 1 which switches produced it.
    "$py $($cmd -join ' ')" | Tee-Object -FilePath $log | Out-Host

    # ForEach-Object turns stderr lines into text, otherwise PowerShell 5.1
    # paints the harmless DirectML warning as a red error block. -u keeps the
    # log readable while the run is going.
    & $py @cmd 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "fold ${f} FEHLGESCHLAGEN ($LASTEXITCODE). Abbruch."
        exit 1
    }

    # The decoupling has to have actually happened. Checking the switch would
    # only check that it was typed; this checks the effect the switch had.
    $text = Get-Content $log -Raw
    if ($text -notmatch 'balance-view at strength') {
        Write-Host "fold ${f}: --balance-view hat nicht gegriffen. Abbruch."
        exit 1
    }
}

Write-Host ""
Write-Host "=== fertig ==="
if ($Rauchtest) {
    Write-Host "Pruefen, ob das Kopffeld ueber dem Lagepriore liegt:"
    Write-Host "  $py rsna\befunde\rsna_kopf_auswertung.py rauchtest --pred-dir $PredDir --folds 0"
    Write-Host ""
    Write-Host "Erst wenn das gruen ist, den langen Lauf starten."
} else {
    Write-Host "Was danach kommt, in dieser Reihenfolge:"
    Write-Host "  $py rsna\befunde\rsna_platt.py"
    Write-Host "      fittet die fuenf Kalibrierkurven und die Schwelle, NUR auf"
    Write-Host "      Entwicklungsdaten, und schreibt serving\model\kalibrierung_p10.json"
    Write-Host "  $py tests\test_rsna_phase10.py"
    Write-Host "      der Waechter, muss gruen sein, bevor der Holdout angefasst wird"
    Write-Host "  $py rsna\pipeline\rsna_holdout.py --dml-index $DmlIndex"
    Write-Host "      der EINE Blick auf die 3812 weggeschlossenen Bilder"
    Write-Host "  $py rsna\befunde\rsna_phase10_auswertung.py"
    Write-Host "      das Urteil"
    Write-Host ""
    Write-Host "ACHTUNG: der Holdout ist in diesem Repo bereits verbraucht."
    Write-Host "rsna_holdout.py bricht deshalb ab. Wer ihn mit --erneut erzwingt,"
    Write-Host "macht aus ihm einen zweiten Selektionssplit."
}
