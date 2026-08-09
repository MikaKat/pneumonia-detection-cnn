<#
  Phase 9, der photometrische Arm. EIN Arm, EIN Hebel.

  WAS GEAENDERT WIRD, UND WAS AUSDRUECKLICH NICHT
  ------------------------------------------------
  Geaendert: --aug-brightness und --aug-contrast gehen von 0,15 auf 0,60.
  Sonst nichts.

  NICHT geaendert: 224 px. Phase 8 hat die Aufloesungsachse geschlossen, und
  die Vorflugmessung sagt, warum dieser Arm gerade dort laufen muss: der
  photometrische Kanal ist bei 224 px am staerksten. Ebenfalls nicht geaendert:
  acht Epochen, Stapel 16, Adapter 1, --balance-view, --head
  --head-negatives exclude, --images data\rsna\png512, --csv data\rsna, und die
  GEOMETRISCHE Augmentierung bleibt bei der alten Staerke (Verschiebung 0,03,
  Skalierung 0,93 bis 1,07). Phase 6 ist durchgefallen, ihr Arm ist nicht der
  Bezug.

  WARUM 0,60 UND NICHT 0,15
  --------------------------
  Vor dem Lauf gemessen mit rsna\befunde\rsna_photometrie_reichweite.py auf
  allen 22872 Entwicklungsbildern. Die Zielmarke stand im Kopf jenes Skripts,
  BEVOR die Zahlen existierten: ein Knopf, der weniger als die Haelfte
  wegnimmt, waere Phase 6 noch einmal.

      globale Helligkeit   AUC -> Projektion 0,4604, Abstand zur Muenze 0,040
      globaler Kontrast    AUC -> Projektion 0,2420, Abstand zur Muenze 0,258

  Der starke globale Kanal ist der KONTRAST, nicht die Helligkeit. Bei 0,15
  nimmt der Jitter 4 bzw. 22 Prozent weg, bei 0,60 sind es 64 und 74.

  DER SCHALTER IST NEU, und das ist der Anlass der halben Phase: bis zum
  09.08.2026 standen brightness und contrast fest verdrahtet mit 0,15 im
  Konstruktor von TrainTransform, waehrend Verschiebung, Skalierung und
  Rotation laengst Argumente waren. Genau diese Gestalt hatte
  --balance-strength, als es gelesen und dann nicht weitergereicht wurde.
  Deshalb schreibt rsna_train.py seit heute nicht nur die beiden Schalter,
  sondern auch die GEZOGENE Staerke in die Ergebniszeile und bricht ab, wenn
  beide auseinandergehen.

  DER VERGLEICHSPARTNER
  ---------------------
  predictions_final_model\, der Gewinner aus Phase 5, derselbe wie in Phase 6, 7
  und 8. Damit sind alle vier Arme auch untereinander lesbar.

  VORFESTLEGUNG, festgelegt VOR diesem Lauf
  ------------------------------------------
  Volltext: erklaerungen\27_phase9_photometrisch.md. Kurz:

  ANKER, beide aus predictions_final_model ueber fuenf Folds:
      A = 0,8368   geschichtete AUC        (erkennt es Pneumonie)
      C = 0,7467   AUC(Score -> Projektion) (verraet es die Aufnahmeart)

  EIN PRIMAERER ENDPUNKT, C. Das obere Ende des gepaarten
      90-Prozent-Intervalls muss unter null liegen. Kein Mindestunterschied,
      genau wie in Phase 6 und 7, damit die vier Arme untereinander lesbar
      bleiben.
  RIEGEL: ein bestandenes Tor zaehlt nur, wenn A nicht unterlegen ist (unteres
      Ende ueber -0,01). Sonst heisst der Satz "ein schlechterer Trenner hat
      weniger zu verraten", und bei einem Jitter dieser Staerke ist das die
      naheliegendste Fehldeutung.
  GRAUZONE: Null im Intervall UND Punktwert auf oder unter -0,015 loest den
      Folgeversuch mit drei Keimen aus, rund 14 h bei 224 px.

  DIE KETTE
  ---------
      .\run_phase9.ps1 -Folds 0     # ein voller Fold, dann hinsehen
      .\run_phase9.ps1              # der Rest, fertige Folds werden uebersprungen
      venv\Scripts\python.exe rsna\befunde\rsna_phase9_auswertung.py

  ZEITEN: bei 224 px kostet ein Fold rund 40 min, fuenf Folds rund 3 h 20. Der
  Jitter selbst kostet nichts, er ist eine Tabellenoperation auf dem PIL-Bild.

  KEIN VERKUERZTER RAUCHTEST, und das ist Absicht. Der Rauchtest hat sich
  zweimal in Folge geirrt, Phase 6 sagte +0,0360 (echt -0,0052), Phase 7 sagte
  +0,0836 (echt +0,0099), beide Male mit best_epoch 1 von 3. Ein Fold bei einem
  Drittel der Epochen ist keine Vorschau.

  WAS NACH FOLD 0 ABGEBROCHEN WERDEN DARF, vorher festgelegt:
      1. eine der Waechterzahlen unten stimmt nicht
      2. Speicherfehler oder Absturz
      3. die Epochenzeit liegt ueber $MAX_EPOCHE_S (dann lief es nicht bei 224)
      4. A auf Fold 0 bricht um mehr als 0,02 ein
  Ein enttaeuschendes dC auf Fold 0 ist AUSDRUECKLICH KEIN Abbruchgrund. Die
  Foldstreuung auf dC liegt bei 0,016 bis 0,025; ein einzelner Fold wuerde den
  Lauf oft allein durch Rauschen abbrechen. In Phase 8 lagen die fuenf
  Foldwerte bei +0,0187, -0,0237, +0,0094, +0,0049 und -0,0069.
#>

param(
    [int[]]$Folds = @(0, 1, 2, 3, 4),
    [int]$DmlIndex = 1,
    [int]$Epochen = 8,
    [int]$CamN = 300
)

# DIE STAERKE IST KEIN PARAMETER, und das ist Absicht.
#
# Dieselbe Falle wie $SIZE in run_phase8.ps1 und $SEITE in run_phase7.ps1: ein
# zweiter Lauf mit -Photo 0.4 haette im Foldloop dieselbe Sprungmarke
# getroffen, fuenfmal "schon fertig, uebersprungen" gemeldet und danach "Alle
# Folds fertig". Beide Varianten teilten sich Tag, Vorhersageordner und
# Logdateien, also genau die Klasse "zwei Arme, ein Tag", die in diesem Projekt
# schon einmal fuenf Gewichte gekostet hat.
#
# Eine andere Staerke ist ein anderer Versuch mit eigener Vorfestlegung. Die
# Reichweitentabelle nennt 0,40 und 0,50 und sagt, was sie kosten.
$PHOTO = 0.60
$SIZE = 224

# Die Epochenzeit bei 224 px, gemessen ueber die Phase-5- bis
# Phase-7-Protokolle: 258 bis 295 s. Hier ist es eine OBERGRENZE und keine
# Untergrenze, und das ist die Umkehrung von Phase 8: dort musste die Epoche
# lang genug sein, um 512 px zu belegen, hier muss sie kurz genug sein, um 512
# px auszuschliessen. Die Zahl ist dieselbe, 2,5 * 262 = 655.
$MAX_EPOCHE_S = 655
$ERWARTET_EPOCHE_S = 275

# KEIN $ErrorActionPreference = "Stop", siehe run_phase5.ps1: PowerShell 5.1
# behandelt sonst die harmlose DirectML-Warnung auf stderr als abbrechenden
# Fehler.
$py = ".\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
New-Item -ItemType Directory -Force -Path logs | Out-Null

# Invariante Kultur fuer JEDE Zahl, die als Argument oder als Suchmuster
# hinausgeht. Auf deutschem Windows macht der Formatoperator -f aus 0.60 die
# Zeichenkette "0,60". Der Waechter faende seine eigene Zeile dann nie und
# braeche einen richtigen Lauf ab. Siehe run_phase6.ps1 bis run_phase8.ps1.
$inv = [cultureinfo]::InvariantCulture
$photoArg = $PHOTO.ToString($inv)
$sizeArg = $SIZE.ToString($inv)
$photoMuster = $PHOTO.ToString("F2", $inv)

$bilder = "data\rsna\png512"
$kaesten = "data\rsna"

# Die erwarteten Protokollzeilen, EINMAL gebaut und an einer Stelle lesbar.
$erwBilder  = "images: " + [regex]::Escape($bilder)
$erwKaesten = "boxes:  " + [regex]::Escape($kaesten)
$erwEingang = "input $SIZE x $SIZE px, measured on the first batch"
# Die ALTE GEOMETRISCHE Augmentierung, ausgeschrieben. Wie in Phase 7 und 8.
$erwAug     = '--aug: rotation 7 deg, translate 0\.030, scale 0\.93 to 1\.07'
# Der Hebel, zweimal. Die erste Zeile ist die Selbstauskunft, die zweite ist
# das, was die Transformation im Loader wirklich gezogen hat. rsna_train.py
# bricht selbst ab, wenn beide auseinandergehen; hier wird nachgelesen, dass es
# die zweite Zeile ueberhaupt gibt. Fehlt sie, ist rsna_train.py aelter als der
# 09.08. und der Hebel ist nirgends belegt.
$erwPhoto   = '--aug photometrisch: brightness ' +
              [regex]::Escape($photoMuster) + ', contrast ' +
              [regex]::Escape($photoMuster)
$erwPhotoGemessen = 'gemessen am Ziehen: brightness '
# Der Beleg aus den DATEN, dass auf dem ganzen Bild gerechnet wurde. Auf dem
# Zuschnitt laege die Kastenabdeckung bei 0,18x.
#
# ACHTUNG: diese Zahl belegt die PHOTOMETRIE NICHT. Sie kann 0,15 und 0,60
# nicht unterscheiden. Der Beleg fuer den Hebel ist $erwPhotoGemessen.
$erwAbdeckung = 'mean tile coverage 0\.11'

$dir = "predictions_p9_photo"
$tag = "_p9photo"
$ergebnis = "results_rsna.csv"
$start = Get-Date

# WANN GILT EIN FOLD ALS FERTIG, und warum ZWEI Bedingungen und nicht eine.
# Wortgleich uebernommen aus run_phase8.ps1: zwischen der Vorhersagedatei und
# der Zeile in results_rsna.csv liegt rund eine Minute, und ein Abbruch in
# diesem Fenster hinterliesse einen Fold MIT Sprungmarke und OHNE
# Ergebniszeile. Der naechste Lauf meldete "schon fertig", und erst die
# Auswertung braeche ab.
function Test-FoldFertig([int]$f) {
    if (-not (Test-Path "$dir\rsna_f${f}_s0.csv")) { return $false }
    if (-not (Test-Path $ergebnis)) { return $false }
    $zeilen = @(Import-Csv $ergebnis | Where-Object {
        $_.tag -eq $tag -and ($_.fold -eq "$f" -or $_.fold -eq "$f.0") })
    return ($zeilen.Count -gt 0)
}

Write-Host "`nPhase 9, photometrischer Jitter $photoArg. Arm '$tag' -> $dir\"
Write-Host "  Bilder $bilder, Kaesten $kaesten, $SIZE px, also das GANZE Bild"
Write-Host "  Folds $($Folds -join ', '), $Epochen Epochen, Adapter $DmlIndex"
Write-Host ""
Write-Host "ERWARTUNG, vorher aufgeschrieben:"
Write-Host "  Epochenzeit rund $ERWARTET_EPOCHE_S s, Obergrenze $MAX_EPOCHE_S s"
Write-Host "  je Fold rund 40 min, fuenf Folds rund 3 h 20"
Write-Host "  gezogene Staerke 0.60 +- 0.15, sonst bricht rsna_train.py ab"
Write-Host "  Kastenabdeckung UNVERAENDERT bei 0.11x (auf dem Zuschnitt: 0.18x)"
Write-Host "  Anker A 0.8368 / C 0.7467 aus predictions_final_model"
Write-Host "  Reichweite des Knopfs, vor dem Lauf gemessen: 64 Prozent des"
Write-Host "    globalen Helligkeitskanals und 74 Prozent des Kontrastkanals weg"
Write-Host ""

# Bevor eine einzige Epoche laeuft: hat rsna_train.py ueberhaupt den Schalter,
# den diese Phase misst, UND misst es die gezogene Staerke nach? Die vier
# Spalten kamen am 09.08. dazu. Ohne sie kann die Auswertung den Hebel spaeter
# nicht belegen, und der Lauf waere dreieinhalb Stunden fuer nichts.
$quelle = Get-Content "rsna\pipeline\rsna_train.py" -Raw
foreach ($spalte in @('"aug_brightness"', '"aug_brightness_measured"')) {
    if ($quelle -notmatch [regex]::Escape($spalte)) {
        Write-Host "ABBRUCH: rsna_train.py schreibt keine Spalte $spalte."
        Write-Host "Diese Fassung ist aelter als der 09.08.2026. Dann steht die"
        Write-Host "photometrische Staerke nirgends im Ergebnis, und ein Arm, der"
        Write-Host "versehentlich bei 0.15 laeuft, saehe wie ein sauberes"
        Write-Host "Nullergebnis aus."
        exit 1
    }
}

# ---- Wachhunde -----------------------------------------------------------
foreach ($t in @("tests\test_rsna_hardware.py", "tests\test_rsna_kopf.py",
                 "tests\test_rsna_phase9.py")) {
    if (-not (Test-Path $t)) { Write-Host "uebersprungen: $t fehlt"; continue }
    Write-Host "`n=== Wachhund: $t ==="
    & $py -u $t 2>&1 | ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath "logs\p9_$([IO.Path]::GetFileNameWithoutExtension($t)).log" |
        Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Pruefungen nicht bestanden. Abbruch."; exit 1
    }
}

foreach ($f in $Folds) {
    if (Test-FoldFertig $f) {
        Write-Host "fold ${f}: schon fertig, uebersprungen"; continue
    }
    if (Test-Path "$dir\rsna_f${f}_s0.csv") {
        Write-Host "fold ${f}: Vorhersagedatei da, aber KEINE Zeile in $ergebnis."
        Write-Host "  Ein frueherer Lauf wurde in dem Fenster zwischen diesen"
        Write-Host "  beiden Schreibvorgaengen abgebrochen. Der Fold wird neu"
        Write-Host "  gerechnet, seine Dateien werden dabei ueberschrieben."
    }
    $log = "logs\p9_photo${photoArg}_f${f}.log"
    Write-Host "`n=== fold ${f}  start $(Get-Date -Format 'HH:mm')  -> $log ==="
    $foldStart = Get-Date

    $cmd = @("-u", "rsna\pipeline\rsna_train.py",
             "--fold", $f, "--epochs", $Epochen, "--batch", 16,
             "--workers", 0, "--cam-n", $CamN,
             "--dml-index", $DmlIndex, "--balance-view",
             "--head", "--head-negatives", "exclude",
             "--size", $sizeArg,
             "--aug-brightness", $photoArg, "--aug-contrast", $photoArg,
             "--images", $bilder, "--csv", $kaesten,
             "--pred-dir", $dir, "--tag", $tag, "--out", $ergebnis)

    "$py $($cmd -join ' ')" | Tee-Object -FilePath $log | Out-Host
    & $py @cmd 2>&1 | ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath $log -Append | Out-Host

    if ($LASTEXITCODE -ne 0) {
        Write-Host "fold ${f} FEHLGESCHLAGEN, Exitcode $LASTEXITCODE. Log: $log"
        Write-Host "Steht dort ABORT mit --aug-brightness, dann ist der Schalter"
        Write-Host "nicht in der Transformation angekommen. NICHT die Schranke"
        Write-Host "aufweiten, sondern die Verkabelung reparieren; genau dafuer"
        Write-Host "ist die Messung da."
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
    if ($text -match 'measurement moved to batch') {
        Write-Host "  Hinweis: erster Stapel ohne Kasten, lambda kam vom naechsten."
    }

    # ---- Die fuenf Waechter DIESER Phase ---------------------------------
    # 1. DER HEBEL, als Schalter.
    if ($text -notmatch $erwPhoto) {
        Write-Host "ABBRUCH: im Protokoll steht nicht die photometrische"
        Write-Host "Staerke $photoMuster. Dieser Arm hat NICHT mit der"
        Write-Host "vorfestgelegten Staerke trainiert."
        exit 1
    }
    # 2. DERSELBE HEBEL, gemessen. Diese Zeile kommt aus den Faktoren, die die
    #    Transformation im Loader wirklich gezogen hat, nicht aus dem Schalter.
    #    rsna_train.py bricht selbst ab, wenn beide auseinandergehen; hier wird
    #    nachgelesen, dass die Zeile ueberhaupt da ist. Fehlt sie, ist der Lauf
    #    mit einer Fassung von rsna_train.py gelaufen, die den Hebel gar nicht
    #    nachmisst, und das ist der teuerste Fehler dieser Phase.
    if ($text -notmatch $erwPhotoGemessen) {
        Write-Host "ABBRUCH: im Protokoll fehlt die GEMESSENE Staerke."
        Write-Host "Dann belegt nur die Selbstauskunft den Hebel, und genau"
        Write-Host "dagegen wurde die Messung gebaut."
        exit 1
    }
    # 3. Die Kantenlaenge, gemessen. Hier ein Abbruchgrund und kein Hebel: die
    #    Aufloesungsachse ist mit Phase 8 geschlossen.
    if ($text -notmatch [regex]::Escape($erwEingang)) {
        Write-Host "ABBRUCH: im Protokoll steht nicht '$erwEingang'."
        Write-Host "Phase 9 laeuft bei $SIZE px in BEIDEN Armen."
        exit 1
    }
    # 4. Die Bilder sind die ORIGINALE, und die GEOMETRIE ist die alte.
    if ($text -notmatch $erwBilder) {
        Write-Host "ABBRUCH: im Protokoll steht nicht 'images: $bilder'."
        exit 1
    }
    if ($text -notmatch $erwKaesten) {
        Write-Host "ABBRUCH: die Kaesten kommen nicht aus '$kaesten'."
        exit 1
    }
    if ($text -notmatch $erwAug) {
        Write-Host "ABBRUCH: die GEOMETRISCHE Augmentierung ist nicht die alte."
        Write-Host "Phase 9 laeuft mit 0.03 / 0.93-1.07. Steht dort die"
        Write-Host "Phase-6-Staerke, bewegen sich zwei Dinge gleichzeitig."
        exit 1
    }
    # 5. Der Beleg aus den DATEN, dass das ganze Bild gemeint war.
    if ($text -notmatch $erwAbdeckung) {
        Write-Host "ABBRUCH: die Kastenabdeckung liegt nicht bei 0.11x."
        Write-Host "Bei rund 0.18 haette der Lauf auf dem ZUSCHNITT gerechnet."
        exit 1
    }

    # ---- Der physikalische Gegentest, umgekehrt zu Phase 8 ---------------
    # Dort musste die Epoche LANG genug sein, um 512 px zu belegen. Hier muss
    # sie KURZ genug sein, um 512 px auszuschliessen. Der Jitter selbst kostet
    # nichts, er ist eine Tabellenoperation auf dem PIL-Bild; wird die Epoche
    # trotzdem lang, lief etwas anderes als geplant.
    $ep = [regex]::Matches($text, 'epoch \d+/\d+.*?\[(\d+)s,')
    if ($ep.Count -eq 0) {
        Write-Host "  Hinweis: keine Epochenzeit im Protokoll gefunden, der"
        Write-Host "  Zeitwaechter faellt fuer diesen Fold aus."
    } else {
        $sek = [int]$ep[0].Groups[1].Value
        Write-Host ("  erste Epoche {0} s (erwartet rund {1} s, Obergrenze {2} s)" `
                    -f $sek, $ERWARTET_EPOCHE_S, $MAX_EPOCHE_S)
        if ($sek -gt $MAX_EPOCHE_S) {
            Write-Host "ABBRUCH: die Epoche lief $sek s, vorfestgelegte"
            Write-Host "Obergrenze ist $MAX_EPOCHE_S s. Bei 224 px kann eine"
            Write-Host "Epoche nicht so lange dauern. Entweder lief der Arm bei"
            Write-Host "einer anderen Kantenlaenge, oder auf der Karte lief noch"
            Write-Host "etwas anderes; im zweiten Fall ist die Zeitmessung"
            Write-Host "unbrauchbar, siehe Phase 8 Fold 0."
            exit 1
        }
    }

    if (-not (Test-Path "$dir\head_f${f}_s0.npz")) {
        Write-Host "ABBRUCH: das Kopffeld wurde nicht geschrieben."; exit 1
    }
    Write-Host ("  fold ${f} fertig in $((Get-Date) - $foldStart)")
    Write-Host ("  Kontrollen: Adapter $DmlIndex, Entkopplung 0.500, Kopf da, " +
                "Photometrie $photoMuster geschaltet UND gemessen, Eingang " +
                "${SIZE}px, ganzes Bild, alte Geometrie, Abdeckung 0.11x, " +
                "Epochenzeit plausibel")
}

Write-Host "`nGesamtdauer $((Get-Date) - $start)"

$fertigeFolds = @(0, 1, 2, 3, 4) | Where-Object { Test-FoldFertig $_ }
Write-Host "fertige Folds in ${dir}: $($fertigeFolds -join ', ')"

if ($fertigeFolds.Count -lt 5) {
    Write-Host "`nNoch nicht alle fuenf Folds da. Das Urteil braucht alle fuenf."
    Write-Host "Weiter mit:  .\run_phase9.ps1"
    Write-Host ""
    Write-Host "Was jetzt angesehen werden darf, und was nicht:"
    Write-Host "  ANSEHEN: die Waechter oben, die gemessene Staerke, die"
    Write-Host "  Epochenzeit, best_epoch, und ob A eingebrochen ist"
    Write-Host "  (Abbruchgrund 4: mehr als 0,02)."
    Write-Host "  NICHT ANSEHEN als Entscheidungsgrundlage: dC auf einem Fold."
    Write-Host "  Die Foldstreuung auf dC liegt bei 0,016 bis 0,025. Ein"
    Write-Host "  einzelner Fold wuerde den Lauf oft allein durch Rauschen"
    Write-Host "  abbrechen. Das steht so in der Vorfestlegung."
} else {
    Write-Host "`nAlle fuenf Folds fertig. Das Urteil:"
    Write-Host "  venv\Scripts\python.exe rsna\befunde\rsna_phase9_auswertung.py"
}
