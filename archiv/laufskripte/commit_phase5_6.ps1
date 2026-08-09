<#
  Nachversionieren: Phase 5, 5b und Phase 6.

  WARUM DAS EIN SKRIPT IST UND KEIN BEFEHL AUS DER SITZUNG
  ---------------------------------------------------------
  Git darf NICHT ueber die Dateibruecke laufen. Zwei unabhaengige Gruende:

    1. Die Bruecke darf schreiben, aber nicht loeschen. Git legt fuer jede
       Schreiboperation .git\index.lock an und loescht sie danach. Das
       Loeschen scheitert, also hinterlaesst schon ein blosses "git status"
       eine tote Sperre.

    2. Am 06.08.2026 dazugekommen und schwerer: auf Windows steht
       core.autocrlf global auf true, im Arbeitsverzeichnis liegen also CRLF
       und im Repo LF. Die Linux-Seite der Bruecke hat eine leere
       Git-Konfiguration und sieht deshalb UEBER HUNDERT Dateien als
       geaendert. Ein Commit von dort wuerde das ganze Repo umschreiben und
       jeden spaeteren Vergleich wertlos machen.

  Deshalb: stagen, ANSEHEN, dann bestaetigen. Das Skript committet nichts,
  bevor du gesehen hast, was drinsteht.

  ES WIRD NICHT GEPUSHT. Das bleibt dein Schritt.

  WAS DAS SKRIPT NICHT ANFASST
  ----------------------------
  logs\*.log         nicht in .gitignore und trotzdem nicht mitgenommen.
                     Ob die Logs ins oeffentliche Repo gehoeren, ist eine
                     Entscheidung und keine Nebenwirkung.
  predictions_p5_rauchtest*\  Rauchtestreste, duerfen weg.
  results_rsna_rauchtest.csv  dito.
  *.pth, *.npz       stehen in .gitignore.
  erklaerungen\      steht in .gitignore.
#>

param([switch]$JaIchWillCommitten)

$py = ".\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

if (-not (Test-Path ".git")) {
    Write-Host "Hier ist kein Repo. Bitte im Projektordner starten."; exit 1
}
if (Test-Path ".git\index.lock") {
    Write-Host "Es liegt eine tote Sperre .git\index.lock."
    Write-Host "Vermutlich hat eine Sitzung ueber die Bruecke Git angefasst."
    Write-Host "  Remove-Item .git\index.lock"
    exit 1
}

# ---- Wachhunde zuerst ----------------------------------------------------
# Was hier hineincommittet wird, soll gruen sein. test_rsna_kopf.py prueft
# unter anderem, dass die neuen Augmentierungsschalter bei ihren Vorgaben
# BITGLEICH das alte Verhalten ziehen. Ohne diesen Nachweis waere der
# Vergleich gegen predictions_final_model nicht mehr sauber.
foreach ($t in @("tests\test_rsna_kopf.py", "tests\test_rsna_detektion.py")) {
    Write-Host "`n=== Wachhund: $t ==="
    & $py -u $t
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n$t nicht bestanden. Nichts committet."; exit 1
    }
}

# ---- Was in welchen Commit gehoert ---------------------------------------
$phase5 = @(
    "rsna\befunde\rsna_lokalisation.py",
    "rsna\befunde\rsna_lokalisation_lesart.py",
    "rsna\befunde\rsna_klassen_kalibrierung.py",
    "rsna\befunde\rsna_phase3_pruefung.py",
    "rsna\befunde\rsna_hardware.py",
    "rsna\befunde\rsna_phase4_pruefung.py",
    "rsna\befunde\rsna_bezugsarm_vergleich.py",
    "rsna\befunde\rsna_kopfraster.py",
    "rsna\befunde\rsna_kopf_auswertung.py",
    "rsna\befunde\rsna_kopf_sel.py",
    "rsna\befunde\rsna_phase5_auswertung.py",
    "rsna\befunde\rsna_phase5b_falschalarm.py",
    "rsna\befunde\rsna_phase5b_detektion.py",
    "rsna\befunde\rsna_cam_power.py",
    "tests\test_rsna_hardware.py",
    "tests\test_rsna_detektion.py",
    "run_phase2.ps1", "run_phase4.ps1", "run_bezugsarm_dml1.ps1",
    "run_phase5.ps1", "run_p5_fold1_neu.ps1",
    "results_rsna.csv",
    "predictions_lokalisation", "predictions_klassen",
    "predictions_hardware", "predictions_kopf",
    "predictions_rsna_dml1",
    "predictions_p5_ref", "predictions_final_model", "predictions_p5_head_em",
    "predictions_p5_auswertung", "predictions_cam_p5",
    "predictions_cam_full", "predictions_cam_smoke",
    "predictions_rsna_base", "predictions_rsna_bal10_s320",
    "archiv\p5_ex_f1_lambda_kaputt"
)
$phase6 = @(
    "rsna\pipeline\rsna_train.py",
    "tests\test_rsna_kopf.py",
    "run_phase6.ps1"
)

$m5 = @"
Phase 5 and 5b: two-headed model, evaluation, detection metric

Adds the analysis scripts written for phases 3 to 5b and the numbers they
produced. The head costs nothing on classification (+0.0081, 90 percent
interval +0.0014 to +0.0149, non-inferiority at margin 0.01) and its field
beats the location prior clearly (point AUC 0.9123 against 0.7520). It does
NOT lower the projection channel, which is what phase 6 and 7 are for.

rsna_phase5b_detektion.py implements the RSNA competition metric (8 IoU
thresholds, greedy matching by confidence) and is checked against a verbatim
copy of the reference implementation on 400 random cases with exact equality.
"@

$m6 = @"
Phase 6: augmentation strength becomes a switch

--aug-translate, --aug-scale and --aug-degrees replace values that were hard
wired in TrainTransform. Their DEFAULTS are the old values, bit identical:
RandomAffine.get_params sees the same three arguments and the random stream
does not depend on a signature. Verified in tests/test_rsna_kopf.py, so the
phase 5 arm stays a valid comparison partner and phase 6 needs no fresh
reference run. The strength is also written to results_rsna.csv as four
columns, so provenance lives in the data and not in a command line.

test_kastenfalle_bei_phase6_staerke checks that image and box mask still move
together at the stronger setting. The counter test shows why this matters
here: two separate draws land 18.74 pixels apart at phase 6 strength against
7.81 at phase 5 strength.
"@

# ---- Stagen, dann ansehen ------------------------------------------------
# Bewusst als Objekte und nicht als verschachtelte Arrays: PowerShells
# Array-Literale und das Entpacken per "$a, $b = $paar" sind an dieser Stelle
# subtil genug, dass ein Fehler darin erst beim Ausfuehren auffiele.
$commits = @(
    [pscustomobject]@{ Name = "Phase 5 und 5b"; Pfade = $phase5; Text = $m5 },
    [pscustomobject]@{ Name = "Phase 6";        Pfade = $phase6; Text = $m6 }
)

foreach ($c in $commits) {
    $name = $c.Name
    $nachricht = $c.Text
    $da  = @($c.Pfade | Where-Object { Test-Path $_ })
    $weg = @($c.Pfade | Where-Object { -not (Test-Path $_) })

    Write-Host "`n`n============================================================"
    Write-Host "  $name"
    Write-Host "============================================================"
    if ($weg.Count -gt 0) {
        Write-Host "nicht vorhanden, uebersprungen: $($weg -join ', ')"
    }
    git add -- $da
    if ($LASTEXITCODE -ne 0) { Write-Host "git add fehlgeschlagen."; exit 1 }

    Write-Host "`nWas jetzt im Index liegt:"
    git diff --cached --stat
    $n = (git diff --cached --name-only | Measure-Object -Line).Lines
    Write-Host "`n$n Dateien."

    if (-not $JaIchWillCommitten) {
        $a = Read-Host "`nCommit '$name' setzen? (j/n)"
        if ($a -ne "j") {
            Write-Host "Abgebrochen. Der Index bleibt stehen, nichts ist verloren."
            Write-Host "Zuruecknehmen mit:  git reset"
            exit 0
        }
    }
    git commit -m $nachricht
    if ($LASTEXITCODE -ne 0) { Write-Host "git commit fehlgeschlagen."; exit 1 }
}

Write-Host "`n`n=== Was danach noch offen ist ==="
git status --short
Write-Host "`nNICHT gepusht. Vorher ansehen:"
Write-Host "  git log --stat -2"
Write-Host "  git ls-files checkpoints\"
Write-Host "Der zweite Befehl ist die Lehre aus dem gescheiterten Deploy:"
Write-Host "das ausgelieferte Gewicht muss im Repo liegen, sonst bricht COPY ab."
