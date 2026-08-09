<#
  Nachversionieren: Phase 7 bis 10 und der Umbau der Webapp.

  Fortsetzung von commit_phase5_6.ps1. Dieselben Gruende, warum das ein Skript
  ist und kein Befehl aus der Sitzung:

    1. Die Dateibruecke darf schreiben, aber nicht loeschen. Git legt fuer jede
       Schreiboperation .git\index.lock an und loescht sie danach. Das Loeschen
       scheitert, also hinterlaesst schon ein blosses "git status" eine tote
       Sperre.

    2. Auf Windows steht core.autocrlf global auf true, im Arbeitsverzeichnis
       liegen also CRLF und im Repo LF. Die Linux-Seite der Bruecke hat eine
       leere Git-Konfiguration und sieht deshalb UEBER HUNDERT Dateien als
       geaendert. Ein Commit von dort wuerde das ganze Repo umschreiben.

  Deshalb: stagen, ANSEHEN, dann bestaetigen. ES WIRD NICHT GEPUSHT.

  NEU GEGENUEBER DEM VORGAENGER: DIE GEWICHTSPRUEFUNG
  ---------------------------------------------------
  Fuenf Gewichte muessen diesmal wirklich ins Repo, weil das Dockerfile sie
  per COPY holt und der Build sonst abbricht. Genau dieser Fall ist am
  26.07.2026 schon einmal eingetreten: eine Zeile *.pth in .gitignore und ein
  COPY, das darauf zeigte.

  .gitignore hat seit dem 09.08.2026 fuer jede der fuenf Dateien eine eigene
  Ausnahme. Das Skript glaubt das nicht, sondern prueft nach dem Stagen mit
  "git ls-files checkpoints\" nach, dass alle fuenf wirklich verfolgt werden,
  und bricht sonst ab. Gesehen ist besser als geglaubt.

  ACHTUNG, GROESSE: die fuenf Gewichte sind zusammen rund 224 MB. Das ist
  unter dem Dateilimit von GitHub (100 MB je Datei) und ueber dessen
  Warnschwelle von 50 MB nicht, aber das Repo waechst dauerhaft um diesen
  Betrag, und Git vergisst nichts. Wer das nicht will, braucht Git LFS, und
  zwar VOR dem ersten Push dieser Dateien.

  ZWEI MELDUNGEN, DIE HARMLOS SIND
  --------------------------------
  "warning: LF will be replaced by CRLF the next time Git touches it"
      Erwartet und richtig so. core.autocrlf steht auf true: im Repo liegt LF,
      im Arbeitsverzeichnis CRLF. Die Dateien, die aus der Sitzung kamen, haben
      LF, und Git sagt nur an, dass es sie beim naechsten Auschecken auf CRLF
      dreht. Es geht dabei nichts verloren.

  Ein Doppelpunkt am Ende einer langen Ausgabe
      Das war Gits Pager, und mit q kommt man heraus. Er ist in diesem Skript
      abgeschaltet (GIT_PAGER=cat weiter unten).

  WAS DAS SKRIPT NICHT ANFASST
  ----------------------------
  logs\*.log         nicht in .gitignore und trotzdem nicht mitgenommen. Ob
                     die Logs ins oeffentliche Repo gehoeren, ist eine
                     Entscheidung und keine Nebenwirkung. Deshalb auch
                     nirgends "git add -A".
  erklaerungen\      steht in .gitignore, bleibt draussen.
  *.npz              steht in .gitignore. Betrifft auch
                     predictions_holdout\holdout_kopffelder.npz, und das
                     ist richtig: die Datei laesst sich neu rechnen.
  data\, venv\       steht in .gitignore.
  *rauchtest*        Reste, duerfen weg.
#>

param([switch]$JaIchWillCommitten)

# Git schickt lange Ausgaben durch einen Pager (less). Der wartet dann auf eine
# Taste, und in einem Skript sieht das aus, als haenge es: unten steht nur ein
# Doppelpunkt. Genau das ist am 09.08.2026 beim ersten Lauf passiert, bei einem
# "git diff --cached --stat" ueber die Vorhersage-CSVs der Phase 6 und 7.
#
# GIT_PAGER=cat schaltet ihn fuer diesen Prozess und alle Git-Aufrufe darin ab.
# Bewusst als Umgebungsvariable und nicht mit --no-pager an jedem einzelnen
# Befehl: ein vergessenes --no-pager waere derselbe Haenger wieder.
$env:GIT_PAGER = "cat"

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
# Was hier hineincommittet wird, soll gruen sein. test_rsna_phase10.py liest
# jede Konstante der Urteilsfunktion gegen erklaerungen\29_phase10_final.md
# nach, damit ein stilles Verschieben im Diff auffaellt.
foreach ($t in @("tests\test_rsna_crops.py", "tests\test_rsna_kopf.py",
                 "tests\test_rsna_phase8.py", "tests\test_rsna_phase9.py",
                 "tests\test_rsna_phase10.py")) {
    if (-not (Test-Path $t)) { Write-Host "uebersprungen, fehlt: $t"; continue }
    Write-Host "`n=== Wachhund: $t ==="
    & $py -u $t
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n$t nicht bestanden. Nichts committet."; exit 1
    }
}

# Der Rauchtest der App braucht pytest und laedt die fuenf Gewichte. Er ist
# der einzige Test, der prueft, dass die App dieselbe Zahl rechnet wie
# rsna_holdout.py. Fehlt pytest, wird das gesagt und nicht verschwiegen.
& $py -c "import pytest" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== Wachhund: tests\test_serving_ensemble.py ==="
    & $py -u -m pytest "tests\test_serving_ensemble.py" -q
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nDer Rauchtest der App ist nicht bestanden. Nichts committet."
        Write-Host "Das heisst: die App rechnet moeglicherweise anders als die"
        Write-Host "Auswertung, deren Zahlen im Ergebnistext stehen."
        exit 1
    }
} else {
    Write-Host "`nACHTUNG: pytest fehlt, tests\test_serving_ensemble.py laeuft NICHT."
    Write-Host "  $py -m pip install pytest"
    Write-Host "Der Commit geht weiter, aber der Abgleich App gegen Auswertung"
    Write-Host "ist damit ungeprueft."
}

# ---- Was in welchen Commit gehoert ---------------------------------------
$gewichte = @(
    "checkpoints\rsna_f0_s0_p5head_ex.pth",
    "checkpoints\rsna_f1_s0_p5head_ex.pth",
    "checkpoints\rsna_f2_s0_p5head_ex.pth",
    "checkpoints\rsna_f3_s0_p5head_ex.pth",
    "checkpoints\rsna_f4_s0_p5head_ex.pth"
)

$phase7 = @(
    "rsna\befunde\rsna_phase6_auswertung.py",
    "rsna\befunde\rsna_phase7_auswertung.py",
    "rsna\befunde\rsna_restkanal.py",
    "rsna\befunde\rsna_crop_qc.py",
    "rsna\pipeline\rsna_crop_masks.py",
    "rsna\pipeline\rsna_make_crops.py",
    "tests\test_rsna_crops.py",
    "archiv\06_augmentierung", "archiv\07_zuschnitt"
)
$phase8 = @(
    "rsna\befunde\rsna_phase8_auswertung.py",
    "tests\test_rsna_phase8.py",
    "archiv\08_aufloesung"
)
$phase9 = @(
    "rsna\befunde\rsna_phase9_auswertung.py",
    "rsna\befunde\rsna_photometrie_reichweite.py",
    "rsna\pipeline\rsna_train.py",
    "rsna\befunde\rsna_cam_power.py",
    "tests\test_rsna_kopf.py", "tests\test_rsna_phase9.py",
    "archiv\09_photometrie",
    "results_rsna.csv"
)
$phase10 = @(
    "rsna\befunde\rsna_platt.py",
    "rsna\befunde\rsna_phase10_auswertung.py",
    "rsna\pipeline\rsna_holdout.py",
    "tests\test_rsna_phase10.py",
    "serving\model\kalibrierung_p10.json",
    "predictions_final_model",
    "predictions_holdout",
    "train_final_model.ps1",
    ".gitignore"
) + $gewichte
$app = @(
    "serving\main.py", "serving\stages.py",
    "serving\model\model.py", "serving\model\__init__.py",
    "serving\Dockerfile", "docker-compose.yml",
    "serving\segmentation",
    "rsna\pipeline\_repo_path.py", "rsna\befunde\_repo_path.py",
    "tests\_repo_path.py",
    "tests\test_serving_ensemble.py",
    "webapp\src\App.jsx",
    "webapp\src\components\HeadField.jsx",
    "webapp\src\components\ResultView.jsx"
)
# Die Umstellung selbst: die uebrigen Archivordner samt ihren Erklaertexten.
# Die Ordner 06 bis 09 haengen an ihren Phasen-Commits oben, damit dort
# Ergebnis und Erklaerung zusammen liegen.
$archiv = @(
    "archiv\README.md",
    "archiv\00_erste_laeufe_und_nebenanalysen",
    "archiv\01_klassen_und_kalibrierung",
    "archiv\02_lokalisation",
    "archiv\03_zweiter_kopf",
    "archiv\04_umgewichtung",
    "archiv\05_hardware",
    "archiv\10_rauchtests",
    "archiv\laufskripte",
    "qc"
)

# Die alten Orte, aus denen im Zuge der Umstellung etwas weggezogen ist. Nur
# fuer die Dateien, die Git bereits kennt: dort muss die Loeschung mit in den
# Index, sonst steht im Repo hinterher beides, der alte und der neue Pfad.
# Fuer alles Uebrige laeuft der Aufruf ins Leere, und das ist in Ordnung.
$alteOrte = @(
    "predictions_rsna", "predictions_rsna_base", "predictions_klassen",
    "predictions_lokalisation", "predictions_lokalisation_fix080",
    "predictions_cam_full", "predictions_cam_p5", "predictions_cam_smoke",
    "predictions_kopf", "predictions_p5_ref", "predictions_p5_head_em",
    "predictions_p5_auswertung", "predictions_rsna_balview",
    "predictions_rsna_bal05", "predictions_rsna_bal10",
    "predictions_rsna_bal10_s320", "predictions_hardware",
    "predictions_rsna_dml1", "predictions_rsna_dml1_rauchtest",
    "predictions_rsna_crop", "predictions_p6_aug", "predictions_p7_fix080",
    "predictions_p8_s512", "predictions_p9_photo",
    "predictions_p5_rauchtest", "predictions_p5_rauchtest_em",
    "predictions_p5_rauchtest_ex", "predictions_p6_rauchtest",
    "predictions_p7_rauchtest_fix080",
    "results_rsna_crop.csv", "results_rsna_rauchtest.csv",
    "run_phase2.ps1", "run_phase4.ps1", "run_phase5.ps1", "run_phase6.ps1",
    "run_phase7.ps1", "run_phase8.ps1", "run_phase9.ps1",
    "run_bal10_night.ps1", "run_baseline_redo_night.ps1",
    "run_bezugsarm_dml1.ps1", "run_night_queue.ps1", "run_p5_fold1_neu.ps1",
    "commit_phase5_6.ps1", "patch_header_measured.py",
    "segmentation", "predictions_p5_head_ex", "predictions_p10_holdout",
    "commit_phase7_10.ps1"
)

$readme = @("README.md")

$m7 = @"
Phase 6 and 7: augmentation and a fixed size crop, both refuted

Two pre-registered five fold arms against the phase 5 anchor (A 0.8368,
C 0.7467), both paired within fold, both failed at their own primary endpoint.

Stronger geometric augmentation moves the projection channel by -0.0052
[-0.0286, +0.0181]. The diagnosis is in the same run: only 24 percent of the
size hint is removed, which is what made the pre flight measurement of phase 9
worth building.

The fixed size crop moves it the wrong way, +0.0099 [-0.0147, +0.0345]. The
side finding is the useful part and rsna_restkanal.py is what measures it: the
crop takes the framing and leaves the anatomy. What the model can see of the
geometry falls from 0.2610 to 0.1638, while the window geometry falls by 77
percent, and the remainder is almost entirely horizontal (width 0.1481 against
height 0.0304). The channel sits on thoracic width, which is partly a finding
in its own right.
"@

$m8 = @"
Phase 8: 512 pixels, failed at both bars

Two equal bars were registered in advance, either stratified AUC up by at
least 0.008 or the projection channel down. Neither happened: A -0.0032, C
+0.0005 [-0.0151, +0.0161]. Both pre registered effect sizes lie OUTSIDE the
measured intervals, which excludes them rather than merely failing to confirm
them.

The finding underneath is that the confounder moved house. At 224 pixels the
readable channel is the global grey value, at 512 it is fine texture, and the
net strength is unchanged. Raising the resolution relocates the shortcut
instead of removing it, so the resolution axis is closed.

The head pools to a fixed 14x14 grid whatever the input size, which is why
this comparison is readable at all: the measuring stick did not move with the
thing it measures.
"@

$m9 = @"
Phase 9: strong photometric jitter, refuted with the lever demonstrated

The first arm in this project where the lever was shown to work BEFORE the run.
rsna_photometrie_reichweite.py measures on all 22872 development images that
the strong global channel is contrast (AUC 0.2420 to projection) and not mean
brightness (0.4604), and that jitter at strength 0.60 leaves about a quarter of
it. The finished models confirm it: sensitivity to a global brightness shift
falls to roughly a quarter.

The projection channel does not fall anyway: -0.0054 [-0.0384, +0.0276]. The
guard on stratified AUC holds at +0.0024. Four arms at the image, four null
results.

The methodological finding is the one that changes future cost. Pairing shrinks
the difference to a third on the classification endpoint in all four arms and
never on the projection endpoint, because the latter is training noise rather
than a property of the fold. More folds do not sharpen it, more seeds do.
"@

$m10 = @"
Phase 10: final model, calibration, and the one pass over the holdout

The deployed model is the ensemble of the five phase 5 fold models, each Platt
calibrated on its own selection split, probabilities averaged, threshold
0.2003. Curves, weight list and threshold live together in
serving/model/kalibrierung_p10.json and come exclusively from development data.

rsna_holdout.py computes the 3812 sealed images once and writes a lock file so
a second pass has to be forced and is then marked as such. The seal was checked
before the first number: no holdout identifier appears in any training or
validation part of any fold.

Result: stratified AUC 0.8687 [0.8566, 0.8805], bar was a lower bound above
0.80. All four pre registered expectations came true. The cross validation was
not optimistic but slightly conservative: the five single models come out
0.0105 higher on the holdout than in cross validation, worst fold -0.0018.

The projection confounder ships with the model at 0.7501 against 0.7467 in
cross validation. Nine phases did not lower it. That is the most important
result of this work and not a blemish.

.gitignore carries an exception for each of the five weights, because the
Dockerfile copies them and a single missing exception breaks the build.
"@

$mApp = @"
Serving: the web app now runs the phase 10 ensemble

Five two headed weights instead of one single headed one, each with its own
Platt curve, probabilities averaged, threshold read from the calibration file
rather than from source. Grad-CAM is the mean of the five maps, because one
fold's map would explain a model that did not produce the number shown. The
head field is drawn as a gradient with no box and no cut off, because its level
is uncalibrated and fires on 62 percent of films without pneumonia.

TwoHeadNet is rebuilt in serving/model/model.py rather than imported from
rsna_train.py, so the serving process does not pull in half the training stack.
The price is a second copy of the class, and tests/test_serving_ensemble.py is
what keeps the two from drifting: it compares parameter names, shapes and
outputs under identical weights, and it checks that the app returns the same
probability for a holdout image as rsna_holdout.py did.

Two defects found while doing this. docker-compose.yml still set THRESHOLD=0.5,
which would have overridden the calibrated threshold in the running container
even though the identical line had been removed from the Dockerfile; main.py
now refuses to start if the variable is set at all. And the app preprocessed
differently from the training loader, converting to greyscale after the resize
instead of before, which is a difference of exactly zero on a grey radiograph
and not zero on a colour upload.

The memory limit in docker-compose.yml goes from 900M to 1600M, because five
ResNet-18 are about 224 MB of parameters.
"@

$mArchiv = @"
Move every experiment that did not ship into archiv/

The main directory held about thirty prediction directories and a dozen run
scripts, which made it impossible to see at a glance what the deployed model
actually consists of. It now holds the model path, the two prediction
directories the shipped model is derived from, and nothing else.

Each archive folder carries a README with the question, what was written down
before the run, the result, and the command to re-run it against the new path.
The per-image predictions move with it, so every number in the main README can
still be recomputed rather than believed.

The analysis scripts stay in rsna/befunde. They import _repo_path from their
own directory, so moving them would have broken them, and their default paths
are all overridable arguments. Each archive README states the call.

Nothing is deleted. archiv/10_rauchtests is the only folder that could be, and
it is kept because two of its runs are the evidence for a rule: a smoke test
that stops at three epochs overestimates the confounder by 0.04 to 0.07, twice
documented.
"@

$mReadme = @"
README: phases 3 to 10

Brings the public README up to the finished project. The headline is no longer
a cross validated number but the holdout: stratified AUC 0.869 on 3812 images
opened once.

Four findings are new. The supervised localisation head is the first map in
this project that beats the anatomical prior (point AUC 0.912 against 0.752),
and it costs nothing on classification. Four pre registered image
interventions all failed at the confounder, and the entry note explains why the
next question of that kind needs seeds rather than folds. Reweighting the
training stream is the only thing that moved it, at a measured price, and is
reported as bought rather than won. The holdout section reports that the cross
validated estimate held.

Removed: the interim two fold status report on the reweighting experiment,
which the five fold result supersedes, and the sentence saying the web app
serves the losing side of that trade, which the rebuild made false.
"@

# ---- Stagen, dann ansehen ------------------------------------------------
$commits = @(
    [pscustomobject]@{ Name = "Phase 6 und 7"; Pfade = $phase7;  Text = $m7 },
    [pscustomobject]@{ Name = "Phase 8";       Pfade = $phase8;  Text = $m8 },
    [pscustomobject]@{ Name = "Phase 9";       Pfade = $phase9;  Text = $m9 },
    [pscustomobject]@{ Name = "Phase 10";      Pfade = $phase10; Text = $m10 },
    [pscustomobject]@{ Name = "Webapp";        Pfade = $app;     Text = $mApp },
    [pscustomobject]@{ Name = "Archiv";        Pfade = $archiv;  Text = $mArchiv },
    [pscustomobject]@{ Name = "README";        Pfade = $readme;  Text = $mReadme }
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

    # Beim Archiv-Commit zusaetzlich die Loeschungen an den alten Orten in den
    # Index nehmen. Ohne das stuende hinterher beides im Repo, der alte und der
    # neue Pfad, und die Verdopplung faellt erst beim Klonen auf. Pfade, die
    # Git ohnehin nie kannte, laufen ins Leere; deshalb wird der Rueckgabewert
    # hier bewusst nicht geprueft.
    if ($name -eq "Archiv") {
        foreach ($alt in $alteOrte) { git add -A -- $alt 2>$null }
        $global:LASTEXITCODE = 0
    }

    # Die Gewichtspruefung gehoert HINTER das Stagen und VOR den Commit. Erst
    # dann sagt "git ls-files" etwas ueber diesen Commit aus.
    if ($name -eq "Phase 10") {
        Write-Host "`n--- Pruefung: liegen alle fuenf Gewichte wirklich im Index? ---"
        $verfolgt = @(git ls-files -- "checkpoints")
        $fehlen = @($gewichte | Where-Object { $verfolgt -notcontains ($_ -replace '\\','/') })
        if ($fehlen.Count -gt 0) {
            Write-Host "ABBRUCH. Diese Gewichte werden nicht verfolgt:"
            $fehlen | ForEach-Object { Write-Host "  $_" }
            Write-Host ""
            Write-Host "Ohne sie liegt das Ensemble nicht im Build-Kontext und der"
            Write-Host "Docker-Build bricht am COPY ab. .gitignore braucht fuer JEDE"
            Write-Host "der fuenf Dateien eine eigene Ausnahme mit fuehrendem !."
            Write-Host "Zuruecknehmen mit:  git reset"
            exit 1
        }
        Write-Host "ok, alle fuenf verfolgt:"
        $verfolgt | Where-Object { $_ -like "*p5head_ex*" } | ForEach-Object { Write-Host "  $_" }

        if (-not (git ls-files -- "serving/model/kalibrierung_p10.json")) {
            Write-Host "`nABBRUCH: serving\model\kalibrierung_p10.json liegt nicht im"
            Write-Host "Index. Ohne sie startet die App nicht: sie fuehrt die fuenf"
            Write-Host "Kurven und die Schwelle."
            exit 1
        }
        Write-Host "ok, die Kalibrierdatei liegt im Index."
    }

    # Nicht "--stat": ein einziger Vorhersageordner sind zwanzig CSV mit
    # zusammen ueber vierzigtausend Zeilen, und die Liste laeuft dann seitenweise
    # durch. Was hier gebraucht wird, ist die Antwort auf "liegt das Richtige
    # drin", und dafuer reichen die Namen und eine Summe.
    $namen = @(git diff --cached --name-only)
    Write-Host "`nWas jetzt im Index liegt ($($namen.Count) Dateien):"
    if ($namen.Count -le 25) {
        $namen | ForEach-Object { Write-Host "  $_" }
    } else {
        # Bei vielen Dateien nach Ordner zusammenfassen, sonst sagt die Liste
        # weniger als ihre eigene Laenge.
        $namen | Group-Object {
            $d = Split-Path $_ -Parent
            if ([string]::IsNullOrEmpty($d)) { "(Wurzel)" } else { $d }
        } | Sort-Object Name | ForEach-Object {
            Write-Host ("  {0,-46} {1,4} Datei(en)" -f $_.Name, $_.Count)
        }
    }
    git diff --cached --shortstat

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
Write-Host "  git --no-pager log --oneline -6"
Write-Host "  git --no-pager ls-files checkpoints\"
Write-Host "(--no-pager, sonst wartet less auf eine Taste und es sieht aus,"
Write-Host " als haenge der Befehl. Mit q kommt man wieder heraus.)"
Write-Host ""
Write-Host "Was in 'git status' stehen BLEIBEN soll:"
Write-Host "  logs\           offene Entscheidung, absichtlich nicht committet"
Write-Host "  *rauchtest*     Reste, duerfen geloescht werden"
Write-Host "  _to_delete\     dito"
Write-Host "erklaerungen\ und *.npz tauchen gar nicht erst auf, sie sind ignoriert."
