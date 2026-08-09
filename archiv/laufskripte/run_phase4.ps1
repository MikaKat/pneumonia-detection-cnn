<#
  Phase 4 der Roadmap: der Hardwarewechsel.

  Die Frage: `torch_directml.device()` ohne Index nimmt Adapter 0, und Adapter 0
  ist auf dieser Maschine die INTEGRIERTE Grafik. Saemtliche Trainings dieses
  Projekts liefen damit auf der APU, nicht auf der RX 5500 XT. Kein Log hat das
  je gesagt, weil `privateuseone:0` die Schnittstelle nennt und nicht den Chip.

  Drei Messungen, alle drei vor dem ersten Lauf im Kopf von
  rsna\befunde\rsna_hardware.py festgeschrieben:

      E1  Geschwindigkeit, Sekunden je Trainingsschritt, in zwei Schleifen
          (nur Chip, und mit echtem Bildpfad). Crossover, mehrere Wiederholungen.
      E2  Gleichheit, dieselben Gewichte und dieselben 64 Bilder durch beide
          Adapter und durch die CPU. Ein Vorwaertsdurchlauf.
      E3  Speicher, 224 bis 512 px bei --batch 16 auf der Steckkarte.

  Kein Training, kein Checkpoint, nichts wird ueberschrieben. Rund fuenf bis
  fuenfzehn Minuten. Ergebnisse landen in predictions_hardware\ und in
  logs\phase4*.log.

  Aufruf, aus dem Repo-Wurzelordner:
      powershell -ExecutionPolicy Bypass -File .\run_phase4.ps1
      powershell -ExecutionPolicy Bypass -File .\run_phase4.ps1 -Schnell
      powershell -ExecutionPolicy Bypass -File .\run_phase4.ps1 -NurBericht

  ERST DEN SCHNELLLAUF, wenn die Karte noch nie gerechnet hat. Er nimmt wenige
  Schritte und eine Wiederholung und zeigt in ein bis zwei Minuten, ob DirectML
  Adapter 1 ueberhaupt annimmt. Erst danach der volle Lauf, dessen Zahlen dann
  auch belastbar sind.

  Der Bericht laesst sich jederzeit ohne Rechnen erzeugen:
      .\venv\Scripts\python.exe rsna\befunde\rsna_hardware.py bericht

  Und die Pruefung, das zweite Skript, das die Schluesse aus den Rohdaten
  nachrechnet und die Fragen stellt, die die Messung selbst nicht stellen kann:
      .\venv\Scripts\python.exe rsna\befunde\rsna_phase4_pruefung.py
#>

param(
    [switch]$Schnell,
    [switch]$NurBericht,
    [int[]]$Adapters = @(0, 1),
    [int]$Speicherkarte = 1
)

$py = ".\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
New-Item -ItemType Directory -Force -Path logs | Out-Null

function Lauf($teil, $argumente, $log) {
    Write-Host "`n=== $teil ==="
    Write-Host "aufruf: $py -u rsna\befunde\rsna_hardware.py $($argumente -join ' ')"
    # ForEach-Object macht aus stderr-Zeilen Text, sonst zeigt PowerShell 5.1
    # harmlose Warnungen als roten Fehlerblock. Out-Host haelt den Ausgabestrom
    # leer, damit $LASTEXITCODE eine Zahl bleibt und kein Array aus dem Log.
    & $py -u rsna\befunde\rsna_hardware.py @argumente 2>&1 |
        ForEach-Object { $_.ToString() } | Tee-Object -FilePath $log | Out-Host
    return $LASTEXITCODE
}

if ($NurBericht) {
    Lauf "Bericht" @("bericht") "logs\phase4_bericht.log" | Out-Null
    exit 0
}

# ---- Vorbedingung: der Wachhund ----------------------------------------
# Die Pruefungen laufen ohne GPU in Sekunden. Faellt eine, ist die Indexuebergabe
# kaputt, und dann misst der Rest dieses Skripts zweimal denselben Chip.
Write-Host "=== Wachhund: tests\test_rsna_hardware.py ==="
& $py -u tests\test_rsna_hardware.py 2>&1 |
    ForEach-Object { $_.ToString() } |
    Tee-Object -FilePath "logs\phase4_tests.log" | Out-Host
if ($LASTEXITCODE -ne 0) {
    Write-Host "Die Pruefungen sind nicht durchgelaufen. Abbruch, denn ohne sie"
    Write-Host "ist nicht gesichert, dass --dml-index ueberhaupt ankommt."
    exit 1
}

$start = Get-Date
Write-Host "`nstart $($start.ToString('yyyy-MM-dd HH:mm'))   python: $py"

# ---- Welche Adapter gibt es ueberhaupt ---------------------------------
if ((Lauf "Adapterliste" @("liste") "logs\phase4_liste.log") -ne 0) {
    Write-Host "torch-directml meldet keinen Adapter. Abbruch."
    exit 1
}

# ---- E1 Geschwindigkeit -------------------------------------------------
$e1 = @("messen", "--adapters") + $Adapters
if ($Schnell) {
    $e1 += @("--steps", 8, "--warmup", 3, "--repeats", 1)
    Write-Host "`nSCHNELLLAUF: wenige Schritte, eine Wiederholung. Diese Zahlen"
    Write-Host "beweisen nur, DASS die Karte rechnet, nicht WIE VIEL schneller."
} else {
    $e1 += @("--steps", 30, "--warmup", 5, "--repeats", 3)
}
$code = Lauf "E1 Geschwindigkeit" $e1 "logs\phase4_e1.log"
if ($code -ne 0) {
    Write-Host "E1 fehlgeschlagen. Haeufigster Grund: Adapter 1 nimmt die"
    Write-Host "Stapelgroesse nicht an. Das ist ein Ergebnis, kein Bedienfehler."
    Write-Host "Weiter mit E3, dort steht es dann als Zahl."
}

# ---- E2 Gleichheit ------------------------------------------------------
$e2 = @("gleich", "--adapters") + $Adapters
Lauf "E2 Gleichheit" $e2 "logs\phase4_e2.log" | Out-Null

# ---- E3 Speicher --------------------------------------------------------
$e3 = @("speicher", "--adapter", $Speicherkarte)
if ($Schnell) { $e3 += @("--steps", 5) }
Lauf "E3 Speicher" $e3 "logs\phase4_e3.log" | Out-Null

# ---- Zusammenfassung ----------------------------------------------------
Lauf "Bericht" @("bericht") "logs\phase4_bericht.log" | Out-Null

# ---- Pruefung, zweites Skript, andere Fragen ----------------------------
# Arbeitsregel des Projekts: die Schluesse einer Phase werden von einem
# ZWEITEN Skript aus den Rohdaten nachgerechnet, bevor die naechste anfaengt.
# Beim Schnelllauf wird sie mitgestartet, meldet dort aber erwartungsgemaess,
# dass eine Wiederholung kein Urteil traegt.
Write-Host "`n=== Pruefung (zweites Skript) ==="
& $py -u rsna\befunde\rsna_phase4_pruefung.py 2>&1 |
    ForEach-Object { $_.ToString() } |
    Tee-Object -FilePath "logs\phase4_pruefung.log" | Out-Host
$pruef = $LASTEXITCODE
if ($pruef -ne 0) {
    Write-Host "`nDie Pruefung hat Befunde. Das ist kein Abbruchgrund, aber sie"
    Write-Host "werden GELESEN, bevor Phase 5 anfaengt. Log: logs\phase4_pruefung.log"
}

Write-Host "`nfertig um $(Get-Date -Format 'HH:mm'), Gesamtdauer $((Get-Date) - $start)"
Write-Host "Ergebnisse: predictions_hardware\bench.csv, gleich.csv, speicher.csv"
Write-Host ""
Write-Host "Was jetzt gilt, und zwar so, wie es VOR den Zahlen aufgeschrieben wurde:"
Write-Host "  E1 bestanden, E2 bestanden, E3 zeigt 224 px  ->  alle neuen Laeufe"
Write-Host "     mit --dml-index 1, und der 224er Bezugsarm wird auf der Karte"
Write-Host "     WIEDERHOLT. Alte APU-Laeufe sind keine gueltigen Partner mehr."
Write-Host "  E1 gefallen, compute aber schneller  ->  der Bildpfad auf der CPU"
Write-Host "     ist die Decke. Naechste Reparatur ist der Bildpfad, nicht der Chip."
Write-Host "  E2 gefallen  ->  Fehlermeldung an torch-directml, kein Wechsel."
Write-Host "  E3 224 px faellt  ->  die Karte ist unbrauchbar. Die Stapelgroesse"
Write-Host "     darf NICHT einfach halbiert werden, ResNet18 hat Batch-Norm."
