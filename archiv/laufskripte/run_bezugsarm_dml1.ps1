<#
  Der Bezugsarm auf der Grafikkarte. Fuenf Folds, 224 px, acht Epochen.

  WARUM
  Phase 4 hat entschieden: ab jetzt laeuft alles auf Adapter 1, der RX 5500 XT.
  Damit sind alle bisherigen Laeufe als Vergleichspartner unbrauchbar, denn sie
  stammen von der integrierten Grafik. Wer den zweiten Kopf aus Phase 5 gegen
  einen APU-Lauf stellt, aendert Hardware und Modellbau gleichzeitig und kann
  das Ergebnis danach keinem von beiden zuschreiben.

  Dieser Lauf ist deshalb keine Wiederholung aus Ordnungsliebe, sondern der
  Bezugsarm, gegen den Phase 5 und alles danach gemessen wird.

  Und er beantwortet nebenbei eine Frage, die bisher nur behauptet war: bewegt
  der Chipwechsel das Ergebnis? Zwei Fuenffachlaeufe, in jedem Schalter gleich,
  verschieden nur im Adapter. Die Auswertung dazu ist vorfestgelegt und steht
  im Kopf von rsna\befunde\rsna_bezugsarm_vergleich.py.

  NICHTS WIRD UEBERSCHRIEBEN
      Gewichte      checkpoints\rsna_f{fold}_s0_dml1.pth   (--tag _dml1)
      Vorhersagen   predictions_rsna_dml1\                 (--pred-dir)
  Der alte Arm in predictions_rsna_base\ bleibt unangetastet, er ist die
  zweite Haelfte des Vergleichs.

  KOSTEN
  Nach der Messung aus Phase 4: rund 255 s je Epoche, acht Epochen, also gut
  34 min je Fold plus Grad-CAM und Stoerungsdurchlaeufe. Fuenf Folds landen bei
  rund dreieinhalb bis vier Stunden. Auf der APU waeren es sieben gewesen.

  AUFRUF, aus dem Repo-Wurzelordner:
      powershell -ExecutionPolicy Bypass -File .\run_bezugsarm_dml1.ps1 -Rauchtest
      powershell -ExecutionPolicy Bypass -File .\run_bezugsarm_dml1.ps1

  ERST DER RAUCHTEST. Er faehrt Fold 0 mit einer Epoche und ohne Grad-CAM,
  rund fuenf Minuten, und schreibt in einen eigenen Ordner. Er beweist, dass
  die Karte einen echten Trainingslauf durchhaelt und dass die Herkunft im Log
  steht. Erst danach der lange Lauf.

  WIEDERAUFNEHMBAR. Ein Fold, dessen Vorhersagedatei schon existiert, wird
  uebersprungen. Der Lauf darf mit Strg+C beendet und spaeter fortgesetzt
  werden.

  Die Auswertung laesst sich jederzeit ohne Rechnen wiederholen:
      .\venv\Scripts\python.exe rsna\befunde\rsna_bezugsarm_vergleich.py
#>

param(
    [switch]$Rauchtest,
    [int[]]$Folds = @(0, 1, 2, 3, 4),
    [int]$DmlIndex = 1
)

$py = ".\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
New-Item -ItemType Directory -Force -Path logs | Out-Null

if ($Rauchtest) {
    $predDir = "predictions_rsna_dml1_rauchtest"
    $tag     = "_dml1_rauchtest"
    $epochen = 1
    $camN    = 0
    $Folds   = @(0)
    # Eigene Ergebnisdatei. Sonst legt der Rauchtest eine Zeile mit
    # dml_index = 1 in results_rsna.csv, und die Herkunftspruefung des
    # Vergleichs zaehlt sie spaeter als Fold des Bezugsarms mit.
    $ergebnis = "results_rsna_rauchtest.csv"
    Write-Host "RAUCHTEST: Fold 0, eine Epoche, kein Grad-CAM, nach $predDir\"
    Write-Host "Er beweist, DASS die Karte trainiert. Die Zahlen daraus sind kein Ergebnis."
} else {
    $predDir  = "predictions_rsna_dml1"
    $tag      = "_dml1"
    $epochen  = 8
    $camN     = 300
    $ergebnis = "results_rsna.csv"
}

# ---- Vorbedingung: Phase 4 muss entschieden sein ------------------------
# Ohne sie ist die Wahl von Adapter 1 nicht belegt, und dieser Lauf waere
# genau der Fehler, den er verhindern soll.
if (-not (Test-Path "predictions_hardware\bench.csv")) {
    Write-Host "Phase 4 ist nicht gemessen (predictions_hardware\bench.csv fehlt)."
    Write-Host "Erst:  powershell -ExecutionPolicy Bypass -File .\run_phase4.ps1"
    exit 1
}

# ---- Vorbedingung: der Wachhund ----------------------------------------
# Sekunden, keine GPU. Faellt er, kommt --dml-index nicht an, und der Lauf
# ginge still auf der APU zu Ende. Das faellt sonst erst nach vier Stunden auf.
Write-Host "=== Wachhund: tests\test_rsna_hardware.py ==="
& $py -u tests\test_rsna_hardware.py 2>&1 |
    ForEach-Object { $_.ToString() } |
    Tee-Object -FilePath "logs\bezugsarm_tests.log" | Out-Host
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pruefungen nicht bestanden. Abbruch."
    exit 1
}

$start = Get-Date
Write-Host "`nstart $($start.ToString('yyyy-MM-dd HH:mm'))  python: $py"
Write-Host "Adapter $DmlIndex, Ziel $predDir\, Tag $tag, $epochen Epochen"
if (-not $Rauchtest) {
    Write-Host "Geschaetzt dreieinhalb bis vier Stunden, also fertig gegen $((Get-Date).AddHours(4).ToString('HH:mm'))."
}

foreach ($f in $Folds) {
    $fertig = "$predDir\rsna_f${f}_s0.csv"
    if (Test-Path $fertig) { Write-Host "fold ${f}: schon fertig, uebersprungen"; continue }

    $log = "logs\bezugsarm_dml1_f${f}.log"
    Write-Host "=== fold ${f}  start $(Get-Date -Format 'HH:mm')  -> $log ==="

    # ForEach-Object macht aus stderr-Zeilen Text, sonst zeigt PowerShell 5.1
    # die harmlose DirectML-Warnung als roten Fehlerblock. -u haelt das Log
    # waehrend des Laufs lesbar.
    & $py -u rsna\pipeline\rsna_train.py --fold $f --epochs $epochen --batch 16 `
        --workers 0 --cam-n $camN --dml-index $DmlIndex `
        --pred-dir $predDir --tag $tag --out $ergebnis 2>&1 |
        ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath $log | Out-Host

    if ($LASTEXITCODE -ne 0) {
        Write-Host "fold ${f} FEHLGESCHLAGEN, Exitcode $LASTEXITCODE. Abbruch. Log: $log"
        break
    }

    # ---- Rauchtest auf die WIRKUNG, nicht auf die Ausgabe ---------------
    # Der teuerste Fehler dieses Projekts war ein Lauf, der etwas anderes tat
    # als er ankuendigte. Deshalb wird das Log auf drei Dinge geprueft.
    $text = Get-Content $log -Raw

    if ($text -notmatch 'Hardware: directml:1') {
        Write-Host "ABBRUCH: im Log steht nicht 'Hardware: directml:1'."
        Write-Host "Der Lauf ging auf den falschen Chip. Alles ab hier waere wertlos."
        break
    }
    Write-Host "  Herkunft: Log nennt Adapter 1, wie es soll"

    if ($text -match 'balance-view at strength') {
        Write-Host "ABBRUCH: dieser Lauf hat umgewichtet, der Bezugsarm darf das nicht."
        break
    }
    Write-Host "  Kontrolle: kein Umgewichtungsblock im Log, wie es sein soll"

    if (-not $Rauchtest -and $text -notmatch 'AUC stratified') {
        Write-Host "WARNUNG: keine geschichtete AUC im Log. Bitte $log ansehen."
    }
}

$da = @($Folds | Where-Object { Test-Path "$predDir\rsna_f${_}_s0.csv" })
Write-Host "`nFolds fertig: $($da.Count) von $($Folds.Count)   Gesamtdauer $((Get-Date) - $start)"

if ($Rauchtest) {
    Write-Host "`nRauchtest durch. Wenn oben 'Herkunft: Log nennt Adapter 1' steht,"
    Write-Host "ist der lange Lauf startklar:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\run_bezugsarm_dml1.ps1"
    Write-Host "Der Rauchtest-Ordner $predDir\ darf danach weg."
    exit 0
}

if ($da.Count -eq $Folds.Count) {
    Write-Host "`n=== Bewegt der Chipwechsel das Ergebnis? ==="
    & $py -u rsna\befunde\rsna_bezugsarm_vergleich.py --karte $predDir 2>&1 |
        ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath "logs\bezugsarm_vergleich.log" | Out-Host
    $code = $LASTEXITCODE
    Write-Host ""
    if ($code -eq 0) {
        Write-Host "Kein Befund. Der Kartenarm ist der gueltige Bezug fuer Phase 5."
    } else {
        Write-Host "Die Auswertung hat Befunde. Lesen, bevor Phase 5 anfaengt."
        Write-Host "Log: logs\bezugsarm_vergleich.log"
    }
} else {
    Write-Host "Vergleich nicht gerechnet, es fehlen Folds. Skript einfach erneut starten."
}
