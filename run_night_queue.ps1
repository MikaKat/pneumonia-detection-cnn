<#
  Two jobs in one queue, weil beide dieselbe Grafikkarte brauchen und deshalb
  nicht gleichzeitig laufen koennen.

  1. Die Basislinie nachbauen. Ihre Gewichte wurden vor dem --tag-Schalter von
     den Zuschnitt- und balance-view-Laeufen ueberschrieben. Die BERICHTETEN
     Zahlen sind davon nicht betroffen, die stehen in predictions_rsna\*.csv
     vom 26.07.; verloren sind nur die Gewichte, und Grad-CAM braucht sie.
     Weil das Training deterministisch ist, muessen die neuen Vorhersagen mit
     den alten uebereinstimmen. Der Vergleich laeuft am Ende automatisch.
     Rund 7 h.

  2. Die Aufloesung anprobieren, ein Fold, gepaart gegen den vorhandenen Lauf
     desselben Folds bei 224 px. Gleiche Variante (volle Entkopplung), gleiche
     acht Epochen, gleicher Seed, gleiche Stapelgroesse. Nur die Kantenlaenge
     aendert sich, sonst waere der Vergleich wertlos.
     320 px rund 2,8 h, 512 px rund 6,7 h.

  Aufruf, aus dem Repo-Wurzelordner:
      powershell -ExecutionPolicy Bypass -File .\run_night_queue.ps1
      powershell -ExecutionPolicy Bypass -File .\run_night_queue.ps1 -Size 512
      powershell -ExecutionPolicy Bypass -File .\run_night_queue.ps1 -SkipResolution

  Wiederaufnehmbar: fertige Folds werden uebersprungen.

  Wenn Schritt 2 mit einem Speicherfehler stirbt, passt 512 px nicht neben der
  Stapelgroesse 16 in den Videospeicher. Dann NICHT einfach den Stapel
  verkleinern und weitermachen: der Vergleich gegen den 224er Lauf waere damit
  unsauber, weil zwei Dinge zugleich anders sind. Stattdessen 320 px versuchen,
  oder den 224er Lauf mit derselben kleineren Stapelgroesse nachziehen.
#>

param(
    [int]$Size = 320,
    [switch]$SkipResolution
)

$py = ".\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
New-Item -ItemType Directory -Force -Path logs | Out-Null

$start = Get-Date
$perFold = @{ 320 = 2.8; 384 = 3.9; 448 = 5.2; 512 = 6.7 }
$est = 7.0 + $(if ($SkipResolution) { 0 } else { $perFold[$Size] })
Write-Host "start $($start.ToString('yyyy-MM-dd HH:mm'))   python: $py"
Write-Host "geschaetzte Dauer $est h, also fertig gegen $($start.AddHours($est).ToString('HH:mm'))"

function Invoke-Run($label, $log, $arguments) {
    Write-Host "=== $label  start $(Get-Date -Format 'HH:mm')  -> $log ==="
    # ForEach-Object macht aus stderr-Zeilen Text: PowerShell 5.1 wuerde die
    # harmlose DirectML-Warnung sonst als roten Fehlerblock zeigen.
    & $py -u rsna\pipeline\rsna_train.py @arguments 2>&1 |
        ForEach-Object { $_.ToString() } | Tee-Object -FilePath $log
    return $LASTEXITCODE
}

# ---- 1. Basislinie nachbauen ------------------------------------------
foreach ($f in 0, 1, 2, 3, 4) {
    if (Test-Path "predictions_rsna_base\rsna_f${f}_s0.csv") {
        Write-Host "Basislinie Fold ${f}: schon fertig, uebersprungen"; continue
    }
    $code = Invoke-Run "Basislinie Fold $f" "logs\base_f${f}.log" @(
        "--fold", $f, "--epochs", 8, "--batch", 16, "--workers", 0,
        "--pred-dir", "predictions_rsna_base", "--tag", "_base")
    if ($code -ne 0) { Write-Host "Fold $f FEHLGESCHLAGEN ($code), Abbruch."; break }
    if ((Get-Content "logs\base_f${f}.log" -Raw) -match 'balance-view at strength') {
        Write-Host "ABBRUCH: dieser Lauf hat umgewichtet, die Basislinie darf das nicht."
        break
    }
    Write-Host "  Pruefung: kein Umgewichtungs-Block im Log, richtig so"
}

$fertig = @(0, 1, 2, 3, 4 | Where-Object { Test-Path "predictions_rsna_base\rsna_f${_}_s0.csv" })
Write-Host "`nBasislinie: $($fertig.Count) von 5 Folds"

if ($fertig.Count -eq 5) {
    Write-Host "=== Reproduzieren die neuen Gewichte die Zahlen vom 26.07.? ==="
    & $py -u -c @"
import pandas as pd, numpy as np
for f in range(5):
    a = pd.read_csv(f'predictions_rsna/rsna_f{f}_s0.csv').set_index('patientId')['p_clean']
    b = pd.read_csv(f'predictions_rsna_base/rsna_f{f}_s0.csv').set_index('patientId')['p_clean']
    c = a.index.intersection(b.index)
    d = float(np.abs(a.loc[c].to_numpy() - b.loc[c].to_numpy()).max())
    print(f'  Fold {f}: n {len(c)}, groesste Abweichung {d:.2e}',
          '  IDENTISCH' if d < 1e-6 else '  ABWEICHEND, nicht verwenden')
"@
}

# ---- 2. Aufloesung anprobieren, ein Fold -------------------------------
if (-not $SkipResolution) {
    $dir = "predictions_rsna_bal10_s$Size"
    if (Test-Path "$dir\rsna_f0_s0.csv") {
        Write-Host "`nAufloesung $Size px Fold 0: schon fertig, uebersprungen"
    } else {
        $code = Invoke-Run "Aufloesung $Size px, Fold 0" "logs\s${Size}_f0.log" @(
            "--fold", 0, "--epochs", 8, "--batch", 16, "--workers", 0,
            "--size", $Size, "--balance-view", "--balance-strength", "1.0",
            "--pred-dir", $dir, "--tag", "_bal10_s$Size")
        if ($code -ne 0) {
            Write-Host "Aufloesungslauf FEHLGESCHLAGEN ($code)."
            Write-Host "Bei einem Speicherfehler: 320 px statt $Size px versuchen."
            Write-Host "Die Stapelgroesse NICHT allein verkleinern, siehe Kopf des Skripts."
        }
    }
}

Write-Host "`nfertig um $(Get-Date -Format 'HH:mm'), Gesamtdauer $((Get-Date) - $start)"
Write-Host "danach:  $py rsna\befunde\rsna_cam_power.py"
