<#
  Phase 2 der Roadmap: die Lokalisation mit dem Messgeraet aus Phase 1 messen.

  Kein Training. Alles laeuft auf der CPU aus vorhandenen Gewichten, die
  Grafikkarte bleibt frei. Gemessen wird auf ALLEN positiven
  Validierungsbildern statt auf 300, in drei Armen:

      base    Basislinie, ganze Bilder, fuenf Folds
      bal10   volle Entkopplung, ganze Bilder, fuenf Folds
      crop    Zuschnitt-Modell, ZUGESCHNITTENE Bilder, vier Folds
              (Fold 0 ist verloren, sein Gewicht wurde ueberschrieben)

  Danebengestellt werden immer Lagepriore, Lungenkarte und Zufall. Erst diese
  vier Zahlen zusammen sagen etwas.

  Aufruf, aus dem Repo-Wurzelordner:
      powershell -ExecutionPolicy Bypass -File .\run_phase2.ps1 -Smoke
      powershell -ExecutionPolicy Bypass -File .\run_phase2.ps1
      powershell -ExecutionPolicy Bypass -File .\run_phase2.ps1 -Arms base,bal10

  ERST DEN RAUCHTEST. `-Smoke` rechnet einen Fold mit zwoelf Bildern je Arm in
  wenigen Minuten und schreibt nach predictions_cam_smoke\. Er beweist, dass
  alle drei Arme laden, dass die Herkunftspruefung anschlaegt und dass die
  Rueckrechnung des Zuschnitt-Arms laeuft. Erst danach der lange Lauf.

  Wiederaufnehmbar: je Fold und Arm eine Datei, fertige Kombinationen werden
  uebersprungen. Der Lauf darf also mit Strg+C beendet und spaeter fortgesetzt
  werden. -Force rechnet alles neu.

  Der Bericht laesst sich jederzeit ohne Rechnen erzeugen:
      .\venv\Scripts\python.exe rsna\befunde\rsna_cam_power.py --report-only
#>

param(
    [switch]$Smoke,
    [switch]$Force,
    [string[]]$Arms = @("base", "bal10", "crop"),
    [int[]]$Folds = @(0, 1, 2, 3, 4)
)

$py = ".\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
New-Item -ItemType Directory -Force -Path logs | Out-Null

# ---- Vorbedingung: die Nulllinien aus Phase 1 --------------------------
# Ohne sie laeuft Phase 2 zwar, aber der Bericht haette keinen Nenner, und
# genau der Nenner ist der Grund fuer diese Phase.
if (-not (Test-Path "predictions_lokalisation\baselines_f4.csv")) {
    Write-Host "=== Phase 1 fehlt, hole sie nach (rund eine Minute) ==="
    & $py -u rsna\befunde\rsna_lokalisation.py tor 2>&1 |
        ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath "logs\lokalisation_tor.log" | Out-Host
    if (-not (Test-Path "predictions_lokalisation\baselines_f4.csv")) {
        Write-Host "Phase 1 ist nicht durchgelaufen, Abbruch."
        exit 1
    }
}

$start = Get-Date
$argumente = @("--arms") + $Arms + @("--folds") + $Folds
if ($Force) { $argumente += "--force" }

if ($Smoke) {
    $log = "logs\phase2_rauchtest.log"
    $argumente = @("--arms") + $Arms + @(
        "--folds", 1, "--n", 12, "--probe", 24,
        "--out-dir", "predictions_cam_smoke")
    if ($Force) { $argumente += "--force" }
    Write-Host "RAUCHTEST, ein Fold, zwoelf Bilder je Arm, nach predictions_cam_smoke\"
} else {
    $log = "logs\phase2.log"
    # Aus dem Rauchtest vom 02.08. hochgerechnet: 36 Karten plus 72
    # Herkunftsdurchlaeufe in rund 6 s echter Rechenzeit. Der volle Lauf sind
    # 14 Kombinationen aus Fold und Arm mal rund 1031 Bilder.
    Write-Host "VOLLER LAUF. Geschaetzt 30 bis 45 min, also fertig gegen $((Get-Date).AddMinutes(45).ToString('HH:mm'))."
    Write-Host "Abbrechen ist erlaubt, fertige Kombinationen bleiben erhalten."
}

Write-Host "start $($start.ToString('yyyy-MM-dd HH:mm'))   python: $py   -> $log"
Write-Host "aufruf: $py -u rsna\befunde\rsna_cam_power.py $($argumente -join ' ')"

# ForEach-Object macht aus stderr-Zeilen Text, sonst zeigt PowerShell 5.1
# harmlose Warnungen als roten Fehlerblock. Out-Host haelt den Ausgabestrom
# leer, damit $LASTEXITCODE eine Zahl bleibt und kein Array aus dem ganzen Log.
& $py -u rsna\befunde\rsna_cam_power.py @argumente 2>&1 |
    ForEach-Object { $_.ToString() } | Tee-Object -FilePath $log | Out-Host
$code = $LASTEXITCODE

Write-Host "`nfertig um $(Get-Date -Format 'HH:mm'), Gesamtdauer $((Get-Date) - $start), Exitcode $code"

if ($code -ne 0) {
    Write-Host "FEHLGESCHLAGEN. Das Log steht in $log."
    Write-Host "Haeufigster Grund: ein Arm findet sein Gewicht nicht, dann steht"
    Write-Host "MISSING im Log und die anderen Arme sind trotzdem gerechnet."
    exit $code
}

if (-not $Smoke) {
    Write-Host "Bericht jederzeit neu: $py rsna\befunde\rsna_cam_power.py --report-only"
    Write-Host "Danach faellig: die Formulierung im README gegen das Ergebnis pruefen."
}
