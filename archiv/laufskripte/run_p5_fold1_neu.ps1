<#
  Fold 1 des Arms `exclude` noch einmal, mit reparierter Lambda-Messung.

  WARUM NUR DIESER EINE FOLD
  --------------------------
  In `exclude` Fold 1 stand lambda bei 75 052 408 statt bei rund 1, weil der
  allererste Stapel kein annotiertes Bild enthielt und der Kopfverlust damit
  exakt null war. Die anderen vierzehn Laeufe der Phase 5 sind davon nicht
  betroffen, ihr lambda liegt zwischen 0,71 und 1,27.

  Die Reparatur in rsna_train.py wartet auf den ersten Stapel MIT einem
  annotierten Bild. Fuer einen Stapel, der schon einen hat, ist der Codepfad
  unveraendert und bitgleich, nachgerechnet in
  `erklaerungen/17_lambda_reparatur.md`. Deshalb bleibt der neue Fold 1 mit den
  vier gesunden Folds vergleichbar, und deshalb muessen die nicht mit
  wiederholt werden.

  WAS DIESES SKRIPT SELBST TUT
  ----------------------------
  Fast nichts, und das ist Absicht. Es prueft zwei Dinge und uebergibt dann an
  `run_phase5.ps1`, das den Lauf schon fuenfzehn Mal richtig gestartet hat.
  Neuer Code, der Dateien verschiebt oder Kommandozeilen zusammenbaut, waere
  hier genau die Sorte ungepruefter Zusatz, an der dieses Projekt schon einmal
  einen Lauf verloren hat.

    1. Ist die Lambda-Reparatur ueberhaupt im Quelltext.
    2. Sind die alten Dateien von Fold 1 aus dem Weg. Sie liegen unter
       archiv\p5_ex_f1_lambda_kaputt\ samt LIESMICH und altem Trainingslog.
       `run_phase5.ps1` ueberspringt jeden Fold, dessen Vorhersagedatei noch
       da ist, deshalb muss das vorher stimmen.

  DAUER
  -----
  Ein Fold, acht Epochen, rund 40 Minuten auf der RX 5500 XT.

  START, aus dem Repo-Wurzelverzeichnis:
      powershell -ExecutionPolicy Bypass -File .\run_p5_fold1_neu.ps1

  Danach, ohne Rechenzeit:
      python rsna\befunde\rsna_phase5_auswertung.py karten --folds 1 --force
      python rsna\befunde\rsna_phase5_auswertung.py bericht
#>

param(
    [int]$Fold = 1
)

$dir = "predictions_final_model"
$vorhersage = "$dir\rsna_f${Fold}_s0.csv"
$archiv = "archiv\p5_ex_f${Fold}_lambda_kaputt"

Write-Host "`nNeulauf exclude Fold $Fold   $(Get-Date -Format 'yyyy-MM-dd HH:mm')"

# ---- 1. Ist die Reparatur drin -------------------------------------------
# Ohne diese Abfrage produzierte der Lauf denselben Fehler noch einmal, und das
# faellt erst nach vierzig Minuten auf.
$src = Get-Content "rsna\pipeline\rsna_train.py" -Raw
if ($src -notmatch 'batch_can_set_lambda') {
    Write-Host "ABBRUCH: rsna_train.py kennt batch_can_set_lambda nicht."
    Write-Host "Die Lambda-Reparatur fehlt. Ohne sie waere dieser Lauf sinnlos."
    exit 1
}
Write-Host "  Reparatur ist im Quelltext"

# ---- 2. Liegt der alte Stand noch im Weg ---------------------------------
if (Test-Path $vorhersage) {
    Write-Host "ABBRUCH: $vorhersage ist noch da."
    Write-Host "run_phase5.ps1 wuerde den Fold als fertig ueberspringen. Den"
    Write-Host "alten Stand nach $archiv verschieben, nicht loeschen: er ist"
    Write-Host "der Beleg fuer den Befund."
    exit 1
}
if (Test-Path $archiv) {
    Write-Host "  alter Stand liegt in $archiv"
} else {
    Write-Host "  HINWEIS: $archiv nicht gefunden. Der alte Lauf ist entweder"
    Write-Host "  woanders abgelegt oder verloren. Vor dem Weitermachen klaeren."
}

# ---- 3. Uebergeben -------------------------------------------------------
Write-Host "`nUebergabe an run_phase5.ps1 -Arme ex -Folds $Fold`n"
& .\run_phase5.ps1 -Arme ex -Folds $Fold
$rc = $LASTEXITCODE
if ($rc -ne 0) {
    Write-Host "`nDer Lauf ist nicht sauber durchgekommen, Exitcode $rc."
    Write-Host "Log: logs\p5_ex_f${Fold}.log"
    exit $rc
}

# ---- 4. Das eine Ergebnis sichtbar machen --------------------------------
# rsna_train.py bricht bei einem lambda ausserhalb der Groessenordnung selbst
# ab. Hier wird der Wert nur noch hingeschrieben, damit niemand ihn aus einem
# Log mit tausend Zeilen suchen muss.
$text = Get-Content "logs\p5_ex_f${Fold}.log" -Raw
$m = [regex]::Match($text, 'lambda measured on batch (\d+):\s+([0-9.eE+-]+)')
if ($m.Success) {
    Write-Host "`n  lambda $($m.Groups[2].Value), gemessen auf Stapel $($m.Groups[1].Value)"
    if ([int]$m.Groups[1].Value -gt 1) {
        Write-Host "  Der erste Stapel hatte wieder keinen Kasten, die Messung"
        Write-Host "  ist weitergewandert. Genau dafuer ist die Reparatur da."
    }
} else {
    Write-Host "`n  WARNUNG: keine Lambda-Zeile im Log gefunden."
}

Write-Host "`nNaechster Schritt, ohne Rechenzeit:"
Write-Host "  python rsna\befunde\rsna_phase5_auswertung.py karten --folds $Fold --force"
Write-Host "  python rsna\befunde\rsna_phase5_auswertung.py bericht"
Write-Host ""
Write-Host "Die Auswertung nimmt die LETZTE Zeile je Arm und Fold aus"
Write-Host "results_rsna.csv und meldet den Wiederholungslauf ausdruecklich."
