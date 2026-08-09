<#
  Phase 7, fester Zuschnitt. EIN Arm, EIN Hebel.

  WAS GEAENDERT WIRD, UND WAS AUSDRUECKLICH NICHT
  ------------------------------------------------
  Geaendert: --images und --csv zeigen auf data\rsna\crop512_fix080 statt auf
  data\rsna\png512 und data\rsna. Sonst nichts.

  NICHT geaendert, und das ist der wichtigste Satz dieser Datei: die
  Augmentierung bleibt bei der ALTEN Staerke, Verschiebung 0,03 und Skalierung
  0,93 bis 1,07. Phase 6 ist am primaeren Endpunkt durchgefallen; ihr Arm ist
  NICHT der Bezug. Liefe Phase 7 mit der Phase-6-Staerke, bewegten sich zwei
  Dinge gleichzeitig, und faellt C, wuesste niemand wovon. Der Waechter unten
  bricht ab, wenn im Protokoll die Phase-6-Staerke steht.

  Ebenfalls nicht geaendert: Aufloesung 224 (das ist Phase 8), acht Epochen,
  Stapel 16, Adapter 1, --balance-view, --head --head-negatives exclude.

  DER VERGLEICHSPARTNER
  ---------------------
  predictions_final_model\, der Gewinner aus Phase 5, derselbe wie in Phase 6.
  Damit sind die beiden Arme auch untereinander lesbar.

  VORFESTLEGUNG, festgelegt VOR diesem Lauf
  ------------------------------------------
  Volltext: erklaerungen\23_phase7_zuschnitt.md. Kurz:

  ANKER, beide aus predictions_final_model ueber fuenf Folds:
      A = 0,8368   geschichtete AUC
      C = 0,7467   AUC(Score -> Projektion)

  PRIMAER: C FAELLT. Gepaart je Fold, 90-Prozent-Intervall, oberes Ende unter
      null.
  NEBENBEDINGUNG: A nicht unterlegen, Marge 0,01, unteres Ende ueber -0,01.
  BEIDES muss halten.

  Die Roadmap sagt hier zum dritten Mal "C muss von 0,8166 fallen" und sieht
  EINEN Fold vor. Beides ist ersetzt: 0,8166 ist Phase 0 ohne Umgewichtung und
  saettigt, und bei einer Foldstreuung auf C von 0,020 entscheidet ein Fold
  nichts.

  AUFLOESUNG, und das gehoert vor den Lauf: gepaart loest dieser Vergleich rund
  0,025 auf. Phase 6 nahm ein Viertel des Groessenhinweises und bewegte C um
  0,005; Phase 7 nimmt 75 Prozent der ganzen Geometrie. Waere die Wirkung
  proportional, laege sie bei 0,016, also UNTER der Aufloesung. Die Grauzone
  ist deshalb ein eigener Ast der Leseregel, mit einer Schwelle, die vorher
  feststeht.

  DIE KETTE
  ---------
      .\run_phase7.ps1 -Schritt zuschnitt   # Bilder erzeugen, CPU
      .\run_phase7.ps1 -Schritt qc          # hinsehen: Kaesten auf den Bildern
      .\run_phase7.ps1 -Schritt priore      # Masken und Lagepriore mitziehen
      .\run_phase7.ps1 -Rauchtest           # Fold 0, drei Epochen, ~20 min
      .\run_phase7.ps1                      # fuenf Folds, rund 3 h 20
      venv\Scripts\python.exe rsna\befunde\rsna_phase7_auswertung.py

  SCHRITT "priore" IST NICHT OPTIONAL. Der Lagepriore aus Phase 1 und die
  Lungenmasken masks224_dev liegen im Koordinatensystem des GANZEN Bildes, das
  Kopffeld dieses Arms im Koordinatensystem des ZUSCHNITTS. Haelt man beides
  gegeneinander, ist der Lagepriore als Grundlinie zu schwach, der Kopf sieht
  besser aus als er ist, und der Rauchtest ginge zu leicht durch. Ein Tor, das
  nur aufgeht, ist kein Tor.
#>

param(
    [int[]]$Folds = @(0, 1, 2, 3, 4),
    [int]$DmlIndex = 1,
    [int]$Epochen = 8,
    [int]$CamN = 300,
    [ValidateSet("zuschnitt", "qc", "priore", "training")]
    [string]$Schritt = "training",
    [switch]$Rauchtest
)

# SEITENLAENGE UND VERSATZ SIND KEINE PARAMETER, und das ist Absicht.
#
# Sie standen zuerst in param(). Beim Gegenlesen am 07.08. fiel auf, dass ein
# zweiter Lauf mit -Seite 0.85 lautlos das Falsche getan haette: Schritt 1 und
# 3 haetten korrekt crop512_fix085 gebaut, im Foldloop haette
# "$dir\rsna_f0_s0.csv" aber die fertigen 0,80-Dateien getroffen, fuenfmal
# "schon fertig, uebersprungen" gemeldet und danach "Alle Folds fertig". Das
# Urteil haette anschliessend ueber die 0,80-Zeilen geurteilt. Dazu haetten
# sich beide Varianten Tag, Vorhersageordner und Logdateien geteilt: genau die
# Klasse "zwei Arme, ein Tag", vor der run_phase5.ps1 warnt und die in diesem
# Projekt schon einmal fuenf Gewichte gekostet hat.
#
# Der Schalter waere auch sachlich falsch. Abschnitt 6 der Vorfestlegung sagt:
# 0,80 steht seit dem 02.08. fest, es wird jetzt nicht daran gedreht. Eine
# andere Seitenlaenge ist ein anderer Versuch mit eigener Vorfestlegung und
# gehoert dann in ein eigenes Skript, nicht in einen Schalter dieses hier.
$SEITE = 0.80
$VERSATZ_Y = 0.03

# KEIN $ErrorActionPreference = "Stop", siehe run_phase5.ps1: PowerShell 5.1
# behandelt sonst die harmlose DirectML-Warnung auf stderr als abbrechenden
# Fehler.
$py = ".\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
New-Item -ItemType Directory -Force -Path logs | Out-Null

# Invariante Kultur fuer JEDE Zahl, die als Argument oder als Suchmuster
# hinausgeht. Auf deutschem Windows macht der Formatoperator -f aus 0.80 die
# Zeichenkette "0,80"; Python schreibt "0.80". Der Waechter faende seine eigene
# Zeile dann nie und braeche einen richtigen Lauf ab. Er versagt zur sicheren
# Seite und kostet trotzdem eine Nacht. Siehe run_phase6.ps1.
$inv = [cultureinfo]::InvariantCulture
$seiteArg = $SEITE.ToString('0.####', $inv)
$versatzArg = $VERSATZ_Y.ToString('0.####', $inv)
$kuerzel = "fix" + ($SEITE * 100).ToString('000', $inv)

$bilder = "data\rsna\crop512_$kuerzel"
$params = "predictions_rsna\crop_params_$kuerzel.csv"
$masken = "data\rsna\masks224_dev_$kuerzel"
$prioreDir = "predictions_lokalisation_$kuerzel"

# Die erwarteten Protokollzeilen, EINMAL gebaut und an einer Stelle lesbar.
# Genau wie $erwartet in run_phase6.ps1: ein Suchmuster, das mitten in einer
# Bedingung zusammengesetzt wird, laesst sich weder lesen noch pruefen.
$erwSeite     = "window: FIXED at " + $SEITE.ToString('0.000', $inv)
$erwBilder    = "images: " + [regex]::Escape($bilder)
$erwKaesten   = "boxes:  " + [regex]::Escape($bilder)
# Die ALTE Augmentierung, ausgeschrieben. Umgekehrt zu Phase 6, und das ist
# der Punkt: dort war die neue Staerke zu belegen, hier die alte.
$erwAug       = '--aug: rotation 7 deg, translate 0\.030, scale 0\.93 to 1\.07'
# Der Beleg aus den Daten. Die Kastenabdeckung geht mit 1/Seite^2, aus 0,117
# werden also 0,18x. 0,117 waere die ALTE Kastentabelle, 0,164 die feste Seite
# 0,85. Das Muster wird aus $SEITE gerechnet und nicht hingeschrieben: eine
# Erwartung, die einer Konstante NEBEN der Konstante steht, geht beim naechsten
# Mal auseinander. Sollwerte je Fold in erklaerungen\23_phase7_zuschnitt.md.
$erwAbdeckung = 'mean tile coverage ' +
    [regex]::Escape((0.1175 / ($SEITE * $SEITE)).ToString('0.00', $inv))

$dir = "predictions_p7_$kuerzel"
$tag = "_p7$kuerzel"
if ($Rauchtest) {
    $Folds = @(0); $Epochen = 3; $CamN = 0
    $dir = "predictions_p7_rauchtest_$kuerzel"; $tag = "_p7rauchtest_$kuerzel"
}
$ergebnis = if ($Rauchtest) { "results_rsna_rauchtest.csv" } else { "results_rsna.csv" }
# Auch die Logdateien tragen, was sie sind. In run_phase6.ps1 schreiben
# Rauchtest und langer Lauf beide nach logs\p6_f0.log, der Rauchtest ist
# also weg, sobald der lange Lauf beginnt: ausgerechnet das Protokoll, gegen
# das man den langen Lauf spaeter haelt.
$laufname = if ($Rauchtest) { "${kuerzel}_rauch" } else { $kuerzel }
$start = Get-Date

# ==========================================================================
# SCHRITT 1: die Bilder
# ==========================================================================
if ($Schritt -eq "zuschnitt") {
    Write-Host "`nPhase 7, Schritt 1: den Zuschnitt erzeugen"
    Write-Host "  feste Seite $seiteArg, Versatz nach unten $versatzArg"
    Write-Host "  -> $bilder"
    Write-Host ""
    Write-Host "DIESE ZAHLEN MUESSEN HERAUSKOMMEN (vorher gerechnet, siehe"
    Write-Host "erklaerungen\23_phase7_zuschnitt.md Abschnitt 11):"
    Write-Host "  Bilder 22872, ohne Maske 0"
    Write-Host "  Seitenlaenge 0.800, Spanne EXAKT 0.000000, Zoom 1.25x"
    Write-Host "  Boxerhalt 0.9961, 54 Bilder unter 90 %, 1 Kasten ganz weg"
    Write-Host "Weicht der Bericht davon ab, nicht weiterrechnen."
    Write-Host ""

    & $py rsna\pipeline\rsna_make_crops.py `
        --ids-from qc\dev_ids.csv `
        --raw-cache data\rsna\unet_raw256.npz `
        --out $bilder --fixed-side $seiteArg --shift-y $versatzArg `
        --params-out $params 2>&1 |
        ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath "logs\p7_${kuerzel}_zuschnitt.log" | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Zuschnitt fehlgeschlagen. Abbruch."; exit 1
    }

    $text = Get-Content "logs\p7_${kuerzel}_zuschnitt.log" -Raw
    if ($text -notmatch [regex]::Escape($erwSeite)) {
        Write-Host "`nABBRUCH: im Bericht steht nicht '$erwSeite'."
        Write-Host "Es wurde eine andere Seitenlaenge zugeschnitten als"
        Write-Host "vorfestgelegt. Vorfestgelegt ist 0,80."
        exit 1
    }
    # Der Rauchtest des festen Fensters. Er haengt an side_ptp (max minus min)
    # und NICHT an side_sd: eine Standardabweichung ueber lauter gleiche Werte
    # ist 1,11e-16 und nicht 0,0, gedruckt sieht sie trotzdem wie 0.000000 aus.
    # Repariert am 07.08. in rsna_make_crops.crop_report.
    if ($text -notmatch 'CONSTANT WINDOW SIZE') {
        Write-Host "`nABBRUCH: der Bericht bestaetigt keine konstante"
        Write-Host "Fenstergroesse. Dann leitet noch ein Pfad die Seitenlaenge"
        Write-Host "aus dem Bild ab, und dieser Lauf misst das ADAPTIVE Fenster"
        Write-Host "unter neuem Namen. Genau das ist am 26.07. schiefgegangen."
        exit 1
    }
    if ($text -notmatch 'leaves the annotated pathology standing') {
        Write-Host "`nABBRUCH: der Zuschnitt schneidet annotierte Pathologie"
        Write-Host "weg. Die AUC-Frage ist dann zweitrangig."
        exit 1
    }
    $n = (Get-ChildItem "$bilder\*.png" | Measure-Object).Count
    Write-Host "`n  $n Bilder in $bilder"
    if ($n -ne 22872) {
        Write-Host "ABBRUCH: erwartet waren 22872 Bilder."; exit 1
    }
    if (-not (Test-Path "$bilder\stage_2_train_labels.csv")) {
        Write-Host "ABBRUCH: die Kastentabelle des Zuschnitts fehlt."; exit 1
    }
    Write-Host "`nSchritt 1 fertig. Weiter mit:  .\run_phase7.ps1 -Schritt qc"
    exit 0
}

# ==========================================================================
# SCHRITT 2: hinsehen
# ==========================================================================
if ($Schritt -eq "qc") {
    Write-Host "`nPhase 7, Schritt 2: die Kaesten auf den zugeschnittenen"
    Write-Host "Bildern ansehen. Kostet Sekunden und faengt einen Fehler, den"
    Write-Host "keine Kennzahl faengt: ein Kasten, der ARITHMETISCH richtig"
    Write-Host "umgerechnet ist und trotzdem am falschen Ort liegt."
    if (-not (Test-Path $params)) {
        Write-Host "ABBRUCH: $params fehlt. Erst -Schritt zuschnitt."; exit 1
    }
    # --images und --csv bleiben auf den ORIGINALEN Vorgaben. Das Skript
    # zeichnet das ganze Bild mit dem Fenster darin und darunter den Ausschnitt;
    # es braucht also das Original und rechnet die Kaesten selbst um. Zeigte man
    # es auf den fertigen Zuschnitt, schnitte es ein zweites Mal.
    # --fixed-side geht mit, obwohl es die gezeichneten Fenster nicht
    # aendert (die kommen aus $params, und deren Mittelpunkt und Groesse sind
    # schon die festen). Es macht die Absicht im Aufruf sichtbar. Die
    # Ueberschrift des Bildes haengt seit 07.08. NICHT mehr daran, sondern an
    # der tatsaechlichen Spanne der Seitenlaengen.
    & $py rsna\befunde\rsna_crop_qc.py --params $params `
        --fixed-side $seiteArg --out "qc\crop_qc_$kuerzel.png" 2>&1 |
        ForEach-Object { $_.ToString() } | Out-Host
    Write-Host "`nBitte qc\crop_qc_$kuerzel.png OEFFNEN. Liegen die Rechtecke"
    Write-Host "auf den Verschattungen, weiter mit:"
    Write-Host "  .\run_phase7.ps1 -Schritt priore"
    exit $LASTEXITCODE
}

# ==========================================================================
# SCHRITT 3: Masken und Lagepriore ins Koordinatensystem des Zuschnitts
# ==========================================================================
if ($Schritt -eq "priore") {
    Write-Host "`nPhase 7, Schritt 3: Lungenmasken und Lagepriore mitziehen"
    if (-not (Test-Path $params)) {
        Write-Host "ABBRUCH: $params fehlt. Erst -Schritt zuschnitt."; exit 1
    }
    & $py rsna\pipeline\rsna_crop_masks.py --params $params `
        --masks data\rsna\masks224_dev --out $masken 2>&1 |
        ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath "logs\p7_${kuerzel}_masken.log" | Out-Host
    if ($LASTEXITCODE -ne 0) { Write-Host "Abbruch."; exit 1 }
    if ((Get-Content "logs\p7_${kuerzel}_masken.log" -Raw) -notmatch 'leaves the lung standing') {
        Write-Host "`nABBRUCH: der Zuschnitt schneidet Lunge weg. Dann heisst"
        Write-Host "'Punkt-AUC innerhalb der Lungenmaske' in den beiden Armen"
        Write-Host "nicht mehr dasselbe, und der Vergleich verliert seine"
        Write-Host "Paarung genau in der Groesse, um die es geht."
        exit 1
    }

    # Nur Fold 0. Der Rauchtest braucht genau prior_f0.npy, und die vier
    # anderen kosten CPU fuer eine Frage, die in dieser Phase niemand stellt:
    # Endpunkt B wird auf dem Zuschnitt nicht gemessen. Wer sie spaeter
    # braucht, ruft denselben Befehl mit --folds 0 1 2 3 4 auf; fertige Folds
    # werden uebersprungen.
    Write-Host "`n--- Lagepriore aus den Kaesten DES ZUSCHNITTS, Fold 0 ---"
    & $py rsna\befunde\rsna_lokalisation.py tor --csv $bilder `
        --masks $masken --out-dir $prioreDir --folds 0 2>&1 |
        ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath "logs\p7_${kuerzel}_priore.log" | Out-Host
    if ($LASTEXITCODE -ne 0) { Write-Host "Abbruch."; exit 1 }
    Write-Host "`nSchritt 3 fertig. Weiter mit:  .\run_phase7.ps1 -Rauchtest"
    exit 0
}

# ==========================================================================
# SCHRITT 4 und 5: Rauchtest und langer Lauf
# ==========================================================================
Write-Host "`nPhase 7, fester Zuschnitt   start $($start.ToString('yyyy-MM-dd HH:mm'))"
Write-Host "Adapter $DmlIndex, Folds: $($Folds -join ', '), Epochen $Epochen"
Write-Host "Bilder und Kaesten: $bilder"
Write-Host "Augmentierung: die ALTE (0.03 / 0.93-1.07), Phase 6 ist nicht der Bezug"
Write-Host "Vergleichspartner: predictions_final_model (A 0,8368, C 0,7467)"
if (-not $Rauchtest) {
    $h = 3.4 * ($Folds.Count / 5.0)
    Write-Host ("Geschaetzt {0:N1} Stunden, also fertig gegen {1}." -f $h,
                (Get-Date).AddHours($h).ToString('dd.MM. HH:mm'))
}

# ---- steht ueberhaupt alles bereit? --------------------------------------
foreach ($p in @($bilder, "$bilder\stage_2_train_labels.csv", $params,
                 $masken, "$prioreDir\prior_f0.npy")) {
    if (-not (Test-Path $p)) {
        Write-Host "`nABBRUCH: $p fehlt."
        Write-Host "Die Kette ist: -Schritt zuschnitt, dann qc, dann priore."
        exit 1
    }
}
Write-Host "  ok  Zuschnitt, Kastentabelle, Masken und Lagepriore sind da"

# ---- Wachhunde -----------------------------------------------------------
# test_rsna_crops.py steht in dieser Phase MIT in der Liste: der Zuschnitt ist
# hier nicht Vorarbeit, sondern der Hebel selbst.
foreach ($t in @("tests\test_rsna_hardware.py", "tests\test_rsna_kopf.py",
                 "tests\test_rsna_crops.py")) {
    Write-Host "`n=== Wachhund: $t ==="
    & $py -u $t 2>&1 | ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath "logs\p7_${laufname}_$([IO.Path]::GetFileNameWithoutExtension($t)).log" |
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
    $log = "logs\p7_${laufname}_f${f}.log"
    Write-Host "`n=== fold ${f}  start $(Get-Date -Format 'HH:mm')  -> $log ==="

    $cmd = @("-u", "rsna\pipeline\rsna_train.py",
             "--fold", $f, "--epochs", $Epochen, "--batch", 16,
             "--workers", 0, "--cam-n", $CamN,
             "--dml-index", $DmlIndex, "--balance-view",
             "--head", "--head-negatives", "exclude",
             "--images", $bilder, "--csv", $bilder,
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
    if ($text -match 'measurement moved to batch') {
        Write-Host "  Hinweis: erster Stapel ohne Kasten, lambda kam vom naechsten."
    }

    # ---- Die drei Waechter DIESER Phase ----------------------------------
    # 1. Der Bildordner. Bis zum 07.08. stand er nirgends im Protokoll; ein Arm
    #    auf png512 haette wie ein sauberes Nullergebnis ausgesehen.
    if ($text -notmatch $erwBilder) {
        Write-Host "ABBRUCH: im Protokoll steht nicht 'images: $bilder'."
        Write-Host "Dieser Arm hat NICHT auf dem Zuschnitt trainiert, oder"
        Write-Host "rsna_train.py ist aelter als der 07.08. und protokolliert"
        Write-Host "den Bildordner gar nicht. Beides macht die Phase wertlos."
        exit 1
    }
    if ($text -notmatch $erwKaesten) {
        Write-Host "ABBRUCH: die Kaesten kommen aus einem anderen Ordner als"
        Write-Host "die Bilder. Der Kopf trainierte dann auf Koordinaten eines"
        Write-Host "anderen Bildes."
        exit 1
    }
    # 2. Die ALTE Augmentierung. Umgekehrt zu Phase 6, und das ist Absicht.
    if ($text -notmatch $erwAug) {
        Write-Host "ABBRUCH: die Augmentierungsstaerke ist nicht die alte."
        Write-Host "Phase 7 laeuft mit 0.03 / 0.93-1.07. Steht dort die"
        Write-Host "Phase-6-Staerke, bewegen sich zwei Dinge gleichzeitig und"
        Write-Host "ein Rueckgang auf C waere keiner Ursache zuzuordnen."
        exit 1
    }
    # 3. Der Beleg aus den DATEN. head_tile_coverage kommt aus den Kaesten, die
    #    der Lauf wirklich geladen hat. Der feste Zuschnitt vergroessert jeden
    #    Kasten um 1/0,80 linear, die Abdeckung muss also von rund 0,117 auf
    #    rund 0,182 steigen. Diese Zeile steht im Protokoll, BEVOR die erste
    #    Epoche laeuft: der teuerste Fehler dieser Phase ist damit in der
    #    ersten halben Minute sichtbar und nicht erst nach dreieinhalb Stunden.
    if ($text -notmatch $erwAbdeckung) {
        Write-Host "ABBRUCH: die Kastenabdeckung liegt nicht bei 0.18x."
        Write-Host "Bei rund 0.117 haette der Lauf die ALTEN Kaesten geladen,"
        Write-Host "bei rund 0.164 waere die feste Seite 0,85 statt 0,80."
        Write-Host "Die Sollwerte je Fold stehen in der Vorfestlegung."
        exit 1
    }

    if (-not (Test-Path "$dir\head_f${f}_s0.npz")) {
        Write-Host "ABBRUCH: das Kopffeld wurde nicht geschrieben."; exit 1
    }
    Write-Host ("  Kontrollen: Adapter $DmlIndex, Entkopplung 0.500, Kopf da, " +
                "Bilder aus dem Zuschnitt, alte Augmentierung, Abdeckung 0.18x")
}

Write-Host "`nGesamtdauer $((Get-Date) - $start)"

if ($Rauchtest) {
    Write-Host "`n=== Kopffeld gegen den Lagepriore, BEIDE im Zuschnitt ==="
    Write-Host "--csv und --masks zeigen auf den Zuschnitt, --baselines auf den"
    Write-Host "dort gerechneten Lagepriore. Mit den Vorgaben verglichen wir"
    Write-Host "zwei Karten aus verschiedenen Koordinatensystemen, und zwar so,"
    Write-Host "dass das Tor zu leicht aufginge."
    & $py rsna\befunde\rsna_kopf_auswertung.py rauchtest --pred-dir $dir `
        --csv $bilder --masks $masken --baselines $prioreDir --folds 0
    $rc = $LASTEXITCODE
    if ($rc -eq 0) {
        Write-Host "`nRauchtest bestanden. Der lange Lauf ist startklar:"
        Write-Host "  powershell -ExecutionPolicy Bypass -File .\run_phase7.ps1"
        Write-Host "$dir\ und checkpoints\*$tag*.pth"
        Write-Host "duerfen danach weg, ebenso results_rsna_rauchtest.csv."
    } else {
        Write-Host "`nRauchtest NICHT bestanden. Den langen Lauf nicht starten."
        Write-Host "Erster Verdaechtiger ist hier NICHT die Kastenfalle,"
        Write-Host "sondern die Umrechnung der Kaesten in das Raster des"
        Write-Host "Zuschnitts. Ansehen: qc\crop_qc_$kuerzel.png, und"
        Write-Host "  $py tests\test_rsna_crops.py"
    }
    exit $rc
}

Write-Host "`nAlle Folds fertig. Naechster Schritt, ohne Rechenzeit:"
Write-Host "  $py rsna\befunde\rsna_phase7_auswertung.py"
Write-Host "Es prueft zuerst, ob dieser Arm den Zuschnitt ueberhaupt gesehen"
Write-Host "hat, und urteilt dann auf C (primaer, muss fallen) und auf A"
Write-Host "(Marge 0,01, darf nicht fallen)."
