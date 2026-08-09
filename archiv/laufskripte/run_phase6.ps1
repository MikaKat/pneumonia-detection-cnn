<#
  Phase 6, Augmentierung. EIN Arm, EIN Hebel.

  WAS GEAENDERT WIRD, UND WAS AUSDRUECKLICH NICHT
  ------------------------------------------------
  Geaendert: Verschiebung von 3 auf 8 Prozent, Skalierung von 0,93 bis 1,07 auf
  0,75 bis 1,00. Sonst nichts. Drehung bleibt bei 7 Grad (mehr ist bei
  Thoraxaufnahmen unphysiologisch), Spiegeln bleibt verboten (Situs inversus,
  widerspricht dem eingedruckten Seitenmarker), Helligkeit und Kontrast bleiben
  bei 0,15.

  NICHT geaendert: das zufaellige Ausradieren, das die Roadmap unter Phase 6
  mit auffuehrt. Es faellt aus diesem Arm heraus, aus zwei Gruenden.

    Erstens beisst es sich mit dem Kopf. Radiert man den Herd weg, sagt das
    Zielfeld weiter "hier ist etwas", und der Kopf wird auf eine Unwahrheit
    trainiert.

    Zweitens, und das wiegt schwerer: fuer das Ausradieren steht in der Roadmap
    keine Frage und kein Endpunkt. Skalierung und Verschiebung haben beides
    (die Fensterrahmung sagt AP gegen PA mit AUC 0,685 vorher, eine kraeftige
    zufaellige Skalierung nimmt ihr die Verlaesslichkeit). Drei Aenderungen in
    einem Arm heissen: faellt C, weiss niemand wovon.

  Das Ausradieren wird ein eigener Arm mit eigener Vorfestlegung, siehe
  erklaerungen\20_phase6_vorfestlegung.md.

  DER VERGLEICHSPARTNER
  ---------------------
  predictions_final_model\, der Gewinner aus Phase 5. Kein neuer Bezugsarm noetig:
  dieser Arm unterscheidet sich von ihm in genau den beiden Zahlen oben. Die
  Schalter --aug-translate und --aug-scale sind bei ihren Vorgaben BITGLEICH
  das, was vorher fest verdrahtet war, nachgerechnet in
  tests\test_rsna_kopf.py.

  VORFESTLEGUNG, festgelegt VOR diesem Lauf
  ------------------------------------------
  Neu verankert, weil die alte Fassung nichts mehr entscheidet. Die Roadmap
  sagt "C muss von 0,8166 fallen". Die 0,8166 sind die Basislinie aus Phase 0
  OHNE Umgewichtung. Seit Phase 5 laeuft jeder Arm mit --balance-view und der
  Anker steht bei C = 0,7467. Jeder Phase-6-Arm waere also schon vor dem Start
  "von 0,8166 gefallen": ein saettigendes Tor, das nichts entscheidet.

  ANKER, beide aus predictions_final_model ueber fuenf Folds:
      A = 0,8368   geschichtete AUC
      C = 0,7467   AUC(Score -> Projektion)

  PRIMAER: C FAELLT. Gepaart je Fold, 90-Prozent-Intervall der Differenz gegen
      den Anker. Das obere Ende muss unter null liegen.

  NEBENBEDINGUNG: A ist nicht unterlegen, Marge 0,01, gepaart je Fold,
      90-Prozent-Intervall, unteres Ende ueber -0,01. Dieselbe Marge wie in
      Phase 5 und Phase 8, also nichts neu erfunden.

  BEIDES muss halten. Faellt C und A bleibt, waere Phase 7 in ihrer teuren Form
  moeglicherweise ueberfluessig, und genau das ist die Frage.

  AUFLOESUNG DES TORES: gepaarte Vergleiche auf C loesten in Phase 5 rund 0,017
  auf (Halbbreite 0,0169 bei ex gegen ref, 0,0147 bei em gegen ref). Ein
  kleinerer Rueckgang wird "unklar" heissen, und unklar heisst zu ungenau
  gemessen, nicht "kein Effekt". Zum Vergleich: die volle Entkopplung bewegte C
  um 0,0233.

  ZUERST DER RAUCHTEST
  --------------------
      powershell -ExecutionPolicy Bypass -File .\run_phase6.ps1 -Rauchtest

  Fold 0, drei Epochen, rund eine Viertelstunde. Er prueft, ob das Kopffeld
  ueber dem Lagepriore liegt. Die Roadmap verlangt ihn hier ausdruecklich: bei
  dieser Augmentierung wirkt ein Fehler in der Mitbewegung der Kaesten viel
  staerker. Gemessen im Gegentest: zwei getrennte Ziehungen liegen bei
  Phase-6-Staerke im Mittel 18,74 Bildpunkte auseinander gegen 7,81 bei
  Phase-5-Staerke.

  DANN DER LAUF
  -------------
      powershell -ExecutionPolicy Bypass -File .\run_phase6.ps1

  Fuenf Folds, rund 3 h 16. Fertige Folds werden uebersprungen.
#>

param(
    [int[]]$Folds = @(0, 1, 2, 3, 4),
    [int]$DmlIndex = 1,
    [int]$Epochen = 8,
    [int]$CamN = 300,
    [double]$Verschiebung = 0.08,
    [double[]]$Skalierung = @(0.75, 1.0),
    [switch]$Rauchtest
)

# KEIN $ErrorActionPreference = "Stop", siehe run_phase5.ps1: PowerShell 5.1
# behandelt sonst die harmlose DirectML-Warnung auf stderr als abbrechenden
# Fehler.
$py = ".\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
New-Item -ItemType Directory -Force -Path logs | Out-Null

$dir = "predictions_p6_aug"
$tag = "_p6aug"
if ($Rauchtest) {
    $Folds = @(0); $Epochen = 3; $CamN = 0
    $dir = "predictions_p6_rauchtest"; $tag = "_p6rauchtest"
}
$ergebnis = if ($Rauchtest) { "results_rsna_rauchtest.csv" } else { "results_rsna.csv" }
$start = Get-Date

# Die erwartete Waechterzeile, EINMAL gebaut und mit INVARIANTER Kultur.
# Warum das hier steht und nicht kurz mit "-f {0:N3}" erledigt wird: der
# Formatoperator -f nimmt die Kultur des angemeldeten Benutzers. Auf einem
# deutschen Windows wird aus 0.08 die Zeichenkette "0,080" mit Komma, waehrend
# Python im Log "0.080" mit Punkt schreibt. Der Wachhund faende seine eigene
# Zeile dann nie und braeche einen voellig richtigen Lauf nach dem ersten Fold
# ab. Er versagt also zur sicheren Seite, aber er kostet eine Nacht.
$inv = [cultureinfo]::InvariantCulture

# Dieselbe Vorsicht auf der Argumentseite. PowerShell wandelt Zahlen fuer
# native Befehle zwar invariant um, sodass argparse "0.75" und nicht "0,75"
# saehe. Aber das ist eine Annahme ueber PowerShell, und sie steht nirgends im
# Skript. Hier steht sie: ausgeschrieben, ueberpruefbar, und die Zeile im Log
# zeigt genau das, was ankommt.
$vArg  = $Verschiebung.ToString('0.####', $inv)
$loArg = $Skalierung[0].ToString('0.####', $inv)
$hiArg = $Skalierung[1].ToString('0.####', $inv)

$erwartet = "--aug: rotation .* translate " +
            [regex]::Escape($Verschiebung.ToString('0.000', $inv)) + ", scale " +
            [regex]::Escape($Skalierung[0].ToString('0.00', $inv)) + " to " +
            [regex]::Escape($Skalierung[1].ToString('0.00', $inv))

Write-Host "`nPhase 6, Augmentierung   start $($start.ToString('yyyy-MM-dd HH:mm'))"
Write-Host "Adapter $DmlIndex, Folds: $($Folds -join ', '), Epochen $Epochen"
Write-Host "Verschiebung $Verschiebung, Skalierung $($Skalierung -join ' bis ')"
Write-Host "Vergleichspartner: predictions_final_model (A 0,8368, C 0,7467)"
if (-not $Rauchtest) {
    $h = 3.3 * ($Folds.Count / 5.0)
    Write-Host ("Geschaetzt {0:N1} Stunden, also fertig gegen {1}." -f $h,
                (Get-Date).AddHours($h).ToString('dd.MM. HH:mm'))
}

# ---- Wachhunde -----------------------------------------------------------
# test_rsna_kopf.py prueft die Kastenfalle jetzt AUCH bei Phase-6-Staerke, und
# das ist der Grund, aus dem er hier vor dem Lauf steht und nicht danach.
foreach ($t in @("tests\test_rsna_hardware.py", "tests\test_rsna_kopf.py")) {
    Write-Host "`n=== Wachhund: $t ==="
    & $py -u $t 2>&1 | ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath "logs\p6_$([IO.Path]::GetFileNameWithoutExtension($t)).log" |
        Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Pruefungen nicht bestanden. Abbruch."; exit 1
    }
}

foreach ($f in $Folds) {
    $fertig = "$dir\rsna_f${f}_s0.csv"
    if (Test-Path $fertig) {
        Write-Host "fold ${f}: schon fertig, uebersprungen"; continue
    }
    $log = "logs\p6_f${f}.log"
    Write-Host "`n=== fold ${f}  start $(Get-Date -Format 'HH:mm')  -> $log ==="

    $cmd = @("-u", "rsna\pipeline\rsna_train.py",
             "--fold", $f, "--epochs", $Epochen, "--batch", 16,
             "--workers", 0, "--cam-n", $CamN,
             "--dml-index", $DmlIndex, "--balance-view",
             "--head", "--head-negatives", "exclude",
             "--aug-translate", $vArg,
             "--aug-scale", $loArg, $hiArg,
             "--pred-dir", $dir, "--tag", $tag, "--out", $ergebnis)

    "$py $($cmd -join ' ')" | Tee-Object -FilePath $log | Out-Host
    & $py @cmd 2>&1 | ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath $log -Append | Out-Host

    if ($LASTEXITCODE -ne 0) {
        Write-Host "fold ${f} FEHLGESCHLAGEN, Exitcode $LASTEXITCODE. Log: $log"
        exit 1
    }

    # ---- Rauchtests auf die WIRKUNG ---------------------------------------
    $text = Get-Content $log -Raw
    if ($text -notmatch "Hardware: directml:$DmlIndex") {
        Write-Host "ABBRUCH: falscher Chip."; exit 1
    }
    if ($text -notmatch 'balance-view at strength') {
        Write-Host "ABBRUCH: kein Umgewichtungsblock im Log."; exit 1
    }
    if ($text -notmatch 'in the stream: 0\.500') {
        Write-Host "ABBRUCH: Entkopplung unvollstaendig."; exit 1
    }
    if ($text -notmatch '--head: 14 x 14 field') {
        Write-Host "ABBRUCH: kein 14 x 14 Kopf im Log."; exit 1
    }
    if ($text -notmatch 'lambda measured on batch') {
        Write-Host "ABBRUCH: lambda wurde nicht gemessen."; exit 1
    }
    # Kein Abbruchgrund, seit die Messung auf den naechsten Stapel wartet. Nur
    # ein Hinweis, damit die verschobene Messung im Protokoll sichtbar bleibt.
    if ($text -match 'measurement moved to batch') {
        Write-Host "  Hinweis: erster Stapel ohne Kasten, lambda kam vom naechsten."
    }
    # Der Waechter DIESER Phase. Ohne ihn koennte der Arm mit der alten
    # Augmentierung durchlaufen und niemand saehe es: die Zahlen faenden sich
    # nur im Rauschen des Bezugsarms wieder.
    if ($text -notmatch $erwartet) {
        Write-Host "ABBRUCH: die Augmentierungsstaerke steht nicht im Log."
        Write-Host "Erwartet: $erwartet"
        Write-Host "Dieser Arm haette mit der Staerke von Phase 5 trainiert."
        exit 1
    }
    # Ohne -not $Rauchtest, genau wie in run_phase5.ps1: der Rauchtest misst
    # das Kopffeld gegen den Lagepriore, er braucht es also erst recht. Hier
    # gemeldet steht der Fehler beim Fold, nicht erst in der Auswertung.
    if (-not (Test-Path "$dir\head_f${f}_s0.npz")) {
        Write-Host "ABBRUCH: das Kopffeld wurde nicht geschrieben."; exit 1
    }
    Write-Host ("  Kontrollen: Adapter $DmlIndex, Entkopplung 0.500, Kopf da, " +
                "Augmentierung wie angekuendigt")
}

Write-Host "`nGesamtdauer $((Get-Date) - $start)"

if ($Rauchtest) {
    Write-Host "`n=== Kopffeld gegen den Lagepriore ==="
    & $py rsna\befunde\rsna_kopf_auswertung.py rauchtest --pred-dir $dir --folds 0
    $rc = $LASTEXITCODE
    if ($rc -eq 0) {
        Write-Host "`nRauchtest bestanden. Der lange Lauf ist startklar:"
        Write-Host "  powershell -ExecutionPolicy Bypass -File .\run_phase6.ps1"
        Write-Host "predictions_p6_rauchtest\ und checkpoints\*_p6rauchtest*.pth"
        Write-Host "duerfen danach weg, ebenso results_rsna_rauchtest.csv."
    } else {
        Write-Host "`nRauchtest NICHT bestanden. Den langen Lauf nicht starten."
        Write-Host "Erster Verdaechtiger bei dieser Augmentierung ist die"
        Write-Host "Kastenfalle: pytest tests\test_rsna_kopf.py"
    }
    exit $rc
}

Write-Host "`nAlle Folds fertig. Naechster Schritt, ohne Rechenzeit: der"
Write-Host "gepaarte Vergleich gegen predictions_final_model auf C (primaer,"
Write-Host "muss fallen) und auf A (Marge 0,01, darf nicht fallen)."
