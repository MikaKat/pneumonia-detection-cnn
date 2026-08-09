<#
  Phase 8, Aufloesung 512 px. EIN Arm, EIN Hebel.

  WAS GEAENDERT WIRD, UND WAS AUSDRUECKLICH NICHT
  ------------------------------------------------
  Geaendert: --size 224 wird zu --size 512. Sonst nichts.

  NICHT geaendert, und das ist der wichtigste Satz dieser Datei: --images und
  --csv bleiben auf data\rsna\png512 und data\rsna, also auf dem GANZEN Bild.
  Der Zuschnitt aus Phase 7 ist hier ausdruecklich nicht dabei. Er traegt bei
  512 px nur 410 echte Bildpunkte (0,80 Seitenlaenge aus einem 512er Original,
  beim Speichern hochgerechnet) und gibt damit keine Aufloesung dazu, sondern
  nur Rahmung. Rahmung hat Phase 7 entlastet. Liefe Phase 8 auf dem Zuschnitt,
  bewegten sich zwei Dinge gleichzeitig, und ein Gewinn waere keiner der beiden
  Aenderungen zuzuordnen.

  Ebenfalls nicht geaendert: acht Epochen, Stapel 16, Adapter 1,
  --balance-view, --head --head-negatives exclude, und die Augmentierung bleibt
  bei der ALTEN Staerke (Verschiebung 0,03, Skalierung 0,93 bis 1,07). Phase 6
  ist durchgefallen, ihr Arm ist nicht der Bezug.

  DER VERGLEICHSPARTNER
  ---------------------
  predictions_final_model\, der Gewinner aus Phase 5, derselbe wie in Phase 6
  und 7. Damit sind alle drei Arme auch untereinander lesbar.

  VORFESTLEGUNG, festgelegt VOR diesem Lauf
  ------------------------------------------
  Volltext: erklaerungen\25_phase8_vorfestlegung.md. Kurz:

  ANKER, beide aus predictions_final_model ueber fuenf Folds:
      A = 0,8368   geschichtete AUC        (erkennt es Pneumonie, soll steigen)
      C = 0,7467   AUC(Score -> Projektion) (verraet es die Aufnahmeart, soll
                                             fallen)

  ZWEI GLEICHRANGIGE TORE. Die Phase zeigt etwas, wenn mindestens eines aufgeht.
      TOR A:  dA >= +0,008 UND unteres Ende des 90-Prozent-Intervalls ueber null.
      TOR C:  oberes Ende des 90-Prozent-Intervalls unter null.
      RIEGEL: ein bestandenes Tor C zaehlt nur, wenn A nicht unterlegen ist
              (unteres Ende ueber -0,01). Sonst heisst der Satz "ein
              schlechterer Trenner hat weniger zu verraten".

  Die Roadmap verlangte auf A einen Mindestunterschied von 0,01. Der wurde VOR
  dem Lauf auf 0,008 gesenkt und die Senkung ist in der Vorfestlegung als
  solche gekennzeichnet: 0,01 liegt ueber jedem A-Gewinn, den dieses Projekt je
  aus einer Bildaenderung gemessen hat (+0,0084 in Phase 7), und ein Tor, das
  nicht aufgehen kann, entscheidet nichts.

  WOHER TOR C KOMMT
  -----------------
  Am 02.08. lief ein einzelner Fold bei 320 px als gepaarter
  Aufloesungsversuch. Berichtet wurde davon nur A (+0,0021, also nichts). Am
  08.08. neu gerechnet fiel C von 0,7381 auf 0,7054, also um 0,0327. Das ist
  der zweitgroesste je gemessene C-Effekt. Ein Fold, ein Keim, aeltere
  Armgeneration: eine Vermutung mit Richtung, kein Ergebnis. Sie stammt aus
  FOLD 0, deshalb berichtet die Auswertung dC ohne Fold 0 daneben.

  DIE KETTE
  ---------
      .\run_phase8.ps1 -Folds 0     # ein voller Fold, dann hinsehen
      .\run_phase8.ps1              # der Rest, fertige Folds werden uebersprungen

  ZEITEN, an Fold 0 gemessen und am 08.08. nach unten korrigiert: ein Fold
  3 h 28, nicht die aus Phase 4 hochgerechneten 1 h 56. Fuenf Folds rund 17 h.
  Siehe den Nachtrag in erklaerungen\25_phase8_vorfestlegung.md.
      venv\Scripts\python.exe rsna\befunde\rsna_phase8_auswertung.py

  KEIN VERKUERZTER RAUCHTEST, und das ist Absicht. Der Rauchtest hat sich
  zweimal in Folge geirrt, Phase 6 sagte +0,0360 (echt -0,0052), Phase 7 sagte
  +0,0836 (echt +0,0099), beide Male mit best_epoch 1 von 3. Ein Fold bei einem
  Drittel der Epochen ist keine Vorschau. Statt dessen laeuft Fold 0 ueber die
  vollen acht Epochen und zaehlt als erster der fuenf mit.

  WAS NACH FOLD 0 ABGEBROCHEN WERDEN DARF, vorher festgelegt:
      1. eine der Waechterzahlen unten stimmt nicht
      2. Speicherfehler oder Absturz
      3. die Epochenzeit liegt unter dem 2,5-fachen der 224er Zeit
      4. A auf Fold 0 bricht um mehr als 0,02 ein
  Ein enttaeuschendes dC auf Fold 0 ist AUSDRUECKLICH KEIN Abbruchgrund. Die
  Foldstreuung auf dC liegt bei 0,0245; ein einzelner Fold wuerde den Lauf etwa
  in der Haelfte der Faelle allein durch Rauschen abbrechen. In Phase 7 lagen
  die fuenf Foldwerte bei -0,0027, -0,0201, -0,0003, +0,0437 und +0,0288.
#>

param(
    [int[]]$Folds = @(0, 1, 2, 3, 4),
    [int]$DmlIndex = 1,
    [int]$Epochen = 8,
    [int]$CamN = 300
)

# DIE KANTENLAENGE IST KEIN PARAMETER, und das ist Absicht.
#
# Dieselbe Falle wie $SEITE in run_phase7.ps1: ein zweiter Lauf mit -Size 384
# haette im Foldloop "$dir\rsna_f0_s0.csv" getroffen, fuenfmal "schon fertig,
# uebersprungen" gemeldet und danach "Alle Folds fertig". Das Urteil haette
# anschliessend ueber die 512er Zeilen geurteilt. Dazu haetten sich beide
# Varianten Tag, Vorhersageordner und Logdateien geteilt: genau die Klasse
# "zwei Arme, ein Tag", die in diesem Projekt schon einmal fuenf Gewichte
# gekostet hat.
#
# Eine andere Kantenlaenge ist ein anderer Versuch mit eigener Vorfestlegung.
$SIZE = 512

# Die 224er Epochenzeit, gemessen ueber die Phase-5- bis Phase-7-Protokolle:
# 258 bis 295 s. Phase 4 hat fuer 512 px die 3,4-fache Schrittzeit gemessen,
# erwartet sind also rund 870 s. Abbruchschwelle ist das 2,5-fache der
# kleinsten beobachteten 224er Zeit, also 2,5 * 262 = 655 s. Darunter ist
# --size nicht dort angekommen, wo es wirkt.
$MIN_EPOCHE_S = 655
# KORRIGIERT am 08.08. NACH Fold 0, und das ist eine Kostenzahl, kein Tor.
# Gemessen wurden 1542 s in der ersten Epoche und danach 1306 bis 1501 s, also
# das 5,1- bis 5,9-fache statt der aus Phase 4 vorhergesagten 3,4. Ein Fold
# kostet damit 3 h 28 statt 1 h 56, fuenf Folds rund 17 h statt 11.
#
# $MIN_EPOCHE_S bleibt bei 655 und wird NICHT nachgezogen: die Untergrenze
# stand vor dem Lauf, sie hat ihre Aufgabe erfuellt, und eine Schwelle, die
# man an das erste Ergebnis anpasst, ist keine Schwelle mehr. Sie trennt 224
# von 512 nach wie vor mit grossem Abstand.
$ERWARTET_EPOCHE_S = 1400

# KEIN $ErrorActionPreference = "Stop", siehe run_phase5.ps1: PowerShell 5.1
# behandelt sonst die harmlose DirectML-Warnung auf stderr als abbrechenden
# Fehler.
$py = ".\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
New-Item -ItemType Directory -Force -Path logs | Out-Null

# Invariante Kultur fuer JEDE Zahl, die als Argument oder als Suchmuster
# hinausgeht. Auf deutschem Windows macht der Formatoperator -f aus 512.0 die
# Zeichenkette "512,0". Der Waechter faende seine eigene Zeile dann nie und
# braeche einen richtigen Lauf ab. Siehe run_phase6.ps1 und run_phase7.ps1.
$inv = [cultureinfo]::InvariantCulture
$sizeArg = $SIZE.ToString($inv)

# Die ORIGINALE, auf beiden Seiten. Umgekehrt zu Phase 7, und das ist der Punkt.
$bilder = "data\rsna\png512"
$kaesten = "data\rsna"

# Die erwarteten Protokollzeilen, EINMAL gebaut und an einer Stelle lesbar.
$erwBilder  = "images: " + [regex]::Escape($bilder)
$erwKaesten = "boxes:  " + [regex]::Escape($kaesten)
# Der Hebel, gemessen statt behauptet. Diese Zeile schreibt rsna_train.py seit
# dem 08.08. aus der Kantenlaenge des ERSTEN Trainingsstapels, nicht aus dem
# Schalter. Ohne sie stuende die Aufloesung in keiner der 76 Spalten von
# results_rsna.csv und in keiner Zeile des Protokolls.
$erwEingang = "input $SIZE x $SIZE px, measured on the first batch"
# Die ALTE Augmentierung, ausgeschrieben. Wie in Phase 7.
$erwAug     = '--aug: rotation 7 deg, translate 0\.030, scale 0\.93 to 1\.07'
# Der Beleg, dass auf dem GANZEN Bild gerechnet wurde. Die Kastenabdeckung ist
# ein Flaechenanteil und haengt fast nicht an der Kantenlaenge: bei 512 px
# aendert sie sich nur um den Faktor 0,99989 bis 1,00016, bleibt also bei 0,11x.
# Auf dem Zuschnitt laege sie bei 0,18x.
#
# ACHTUNG, das ist wichtig und steht deshalb hier: diese Zahl belegt die
# AUFLOESUNG NICHT. Sie kann 224 und 512 nicht unterscheiden. Der Beleg fuer
# den Hebel ist $erwEingang, nicht diese Zeile. Sollwerte je Fold in
# erklaerungen\25_phase8_vorfestlegung.md.
$erwAbdeckung = 'mean tile coverage 0\.11'

$dir = "predictions_p8_s${SIZE}"
$tag = "_p8s${SIZE}"
$ergebnis = "results_rsna.csv"
$start = Get-Date

# WANN GILT EIN FOLD ALS FERTIG, und warum ZWEI Bedingungen und nicht eine.
#
# rsna_train.py schreibt am Ende eines Folds in dieser Reihenfolge:
#     1. predictions_p8_s512\rsna_f{N}_s0.csv    <-- bisher die Sprungmarke
#     2. sel_..., cam_..., head_....npz
#     3. checkpoints\..._p8s512.pth
#     4. die Zeile in results_rsna.csv           <-- rund eine Minute spaeter
#
# Zwischen 1 und 4 liegt ein Fenster von etwa einer Minute. Ein Abbruch darin,
# durch Strg+C, Stromausfall oder Neustart, hinterliesse einen Fold MIT
# Sprungmarke und OHNE Ergebniszeile. Mit der alten Bedingung meldete der
# naechste Lauf "schon fertig, uebersprungen" und danach "Alle fuenf Folds
# fertig". Erst die Auswertung braeche ab, mit "die Arme decken verschiedene
# Folds ab", und der Fold muesste nach vierzehn Stunden Wartezeit nachgerechnet
# werden.
#
# Das ist dieselbe Klasse wie der -Size-Schalter weiter oben: eine Sprungmarke,
# die weniger prueft, als sie behauptet. Deshalb hier zwei Bedingungen.
#
# Fehlt die Zeile, wird der Fold NEU gerechnet und nicht abgebrochen.
# rsna_train.py ueberschreibt dabei alle seine Ausgabedateien, es geht also
# nichts verloren; ein zweiter Lauf kann hoechstens eine doppelte Zeile
# erzeugen, und die faengt zeilen_holen in der Auswertung ab.
#
# Die Abfrage auf "$f.0" steht daneben, weil results_rsna.csv gelesen, um eine
# Zeile ergaenzt und neu geschrieben wird: sobald eine Spalte in irgendeiner
# Zeile leer ist, macht pandas eine Fliesskommaspalte daraus, und aus 0 wird
# 0.0. Bei `fold` ist das heute nicht so (geprueft am 08.08., die Werte stehen
# als '0' bis '4' in der Datei), bei `size` und `dml_index` schon. Eine
# Erwartung, die nur fuer den heutigen Zustand stimmt, faellt beim naechsten
# Mal um.
#
# Geprueft am 08.08. unter PowerShell 7.4 gegen die echte results_rsna.csv,
# einschliesslich des NUL-Bytes, das torch-directml in device_name
# hinterlaesst: Import-Csv liest die Datei vollstaendig, 54 Zeilen.
function Test-FoldFertig([int]$f) {
    if (-not (Test-Path "$dir\rsna_f${f}_s0.csv")) { return $false }
    if (-not (Test-Path $ergebnis)) { return $false }
    $zeilen = @(Import-Csv $ergebnis | Where-Object {
        $_.tag -eq $tag -and ($_.fold -eq "$f" -or $_.fold -eq "$f.0") })
    return ($zeilen.Count -gt 0)
}

Write-Host "`nPhase 8, Aufloesung $SIZE px. Arm '$tag' -> $dir\"
Write-Host "  Bilder $bilder, Kaesten $kaesten, also das GANZE Bild"
Write-Host "  Folds $($Folds -join ', '), $Epochen Epochen, Adapter $DmlIndex"
Write-Host ""
Write-Host "ERWARTUNG, vorher aufgeschrieben:"
Write-Host "  Epochenzeit rund $ERWARTET_EPOCHE_S s (bei 224 px sind es 262 s)"
Write-Host "  je Fold rund 3 h 28, fuenf Folds rund 17 h (an Fold 0 gemessen)"
Write-Host "  Kastenabdeckung UNVERAENDERT bei 0.11x (auf dem Zuschnitt: 0.18x)"
Write-Host "  Anker A 0.8368 / C 0.7467 aus predictions_final_model"
Write-Host ""

# Bevor eine einzige Epoche laeuft: hat rsna_train.py ueberhaupt den Schalter,
# den diese Phase misst? Die zwei Spalten kamen am 08.08. dazu. Ohne sie kann
# die Auswertung den Hebel spaeter nicht belegen, und der Lauf waere elf
# Stunden fuer nichts.
$quelle = Get-Content "rsna\pipeline\rsna_train.py" -Raw
if ($quelle -notmatch '"input_px"') {
    Write-Host "ABBRUCH: rsna_train.py schreibt keine Spalte input_px."
    Write-Host "Diese Fassung ist aelter als der 08.08.2026. Dann steht die"
    Write-Host "Aufloesung nirgends im Ergebnis, und ein Arm, der versehentlich"
    Write-Host "bei 224 px laeuft, saehe wie ein sauberes Nullergebnis aus."
    exit 1
}

# ---- Wachhunde -----------------------------------------------------------
foreach ($t in @("tests\test_rsna_hardware.py", "tests\test_rsna_kopf.py",
                 "tests\test_rsna_phase8.py")) {
    if (-not (Test-Path $t)) { Write-Host "uebersprungen: $t fehlt"; continue }
    Write-Host "`n=== Wachhund: $t ==="
    & $py -u $t 2>&1 | ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath "logs\p8_$([IO.Path]::GetFileNameWithoutExtension($t)).log" |
        Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Pruefungen nicht bestanden. Abbruch."; exit 1
    }
}

foreach ($f in $Folds) {
    if (Test-FoldFertig $f) {
        Write-Host "fold ${f}: schon fertig, uebersprungen"; continue
    }
    # Der abgebrochene Fold: Datei da, Zeile nicht. Siehe Test-FoldFertig.
    if (Test-Path "$dir\rsna_f${f}_s0.csv") {
        Write-Host "fold ${f}: Vorhersagedatei da, aber KEINE Zeile in $ergebnis."
        Write-Host "  Ein frueherer Lauf wurde in dem Fenster zwischen diesen"
        Write-Host "  beiden Schreibvorgaengen abgebrochen. Der Fold wird neu"
        Write-Host "  gerechnet, seine Dateien werden dabei ueberschrieben."
    }
    $log = "logs\p8_s${SIZE}_f${f}.log"
    Write-Host "`n=== fold ${f}  start $(Get-Date -Format 'HH:mm')  -> $log ==="
    $foldStart = Get-Date

    $cmd = @("-u", "rsna\pipeline\rsna_train.py",
             "--fold", $f, "--epochs", $Epochen, "--batch", 16,
             "--workers", 0, "--cam-n", $CamN,
             "--dml-index", $DmlIndex, "--balance-view",
             "--head", "--head-negatives", "exclude",
             "--size", $sizeArg,
             "--images", $bilder, "--csv", $kaesten,
             "--pred-dir", $dir, "--tag", $tag, "--out", $ergebnis)

    "$py $($cmd -join ' ')" | Tee-Object -FilePath $log | Out-Host
    & $py @cmd 2>&1 | ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath $log -Append | Out-Host

    if ($LASTEXITCODE -ne 0) {
        Write-Host "fold ${f} FEHLGESCHLAGEN, Exitcode $LASTEXITCODE. Log: $log"
        Write-Host "Bei einem Speicherfehler: NICHT den Stapel verkleinern und"
        Write-Host "weitermachen. Der Vergleich gegen den 224er Arm waere damit"
        Write-Host "unsauber, weil zwei Dinge zugleich anders sind. Statt dessen"
        Write-Host "den Bezugsarm mit derselben kleineren Stapelgroesse nachziehen"
        Write-Host "oder auf 448 px gehen, beides mit eigener Vorfestlegung."
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

    # ---- Die vier Waechter DIESER Phase ----------------------------------
    # 1. DER HEBEL, gemessen. Diese Zeile steht im Protokoll, BEVOR die erste
    #    Epoche laeuft, und sie kommt aus dem Tensor und nicht aus dem
    #    Schalter. rsna_train.py bricht selbst ab, wenn beide nicht
    #    uebereinstimmen; hier wird nachgelesen, dass die Zeile ueberhaupt da
    #    ist. Der teuerste Fehler dieser Phase ist damit in der ersten halben
    #    Minute sichtbar und nicht erst nach zwei Stunden.
    if ($text -notmatch [regex]::Escape($erwEingang)) {
        Write-Host "ABBRUCH: im Protokoll steht nicht '$erwEingang'."
        Write-Host "Dieser Arm hat NICHT bei $SIZE px trainiert, oder"
        Write-Host "rsna_train.py ist aelter als der 08.08. und misst die"
        Write-Host "Kantenlaenge gar nicht. Beides macht die Phase wertlos."
        exit 1
    }
    # 2. Die Bilder sind die ORIGINALE. Umgekehrt zu Phase 7: dort war der
    #    Unterschied dieser beiden Pfade der Hebel, hier waere er ein zweiter
    #    Unterschied und damit das Ende des gepaarten Vergleichs.
    if ($text -notmatch $erwBilder) {
        Write-Host "ABBRUCH: im Protokoll steht nicht 'images: $bilder'."
        Write-Host "Phase 8 laeuft auf dem GANZEN Bild. Ein Arm auf dem"
        Write-Host "Zuschnitt aenderte zwei Dinge auf einmal."
        exit 1
    }
    if ($text -notmatch $erwKaesten) {
        Write-Host "ABBRUCH: die Kaesten kommen nicht aus '$kaesten'."
        exit 1
    }
    # 3. Die ALTE Augmentierung.
    if ($text -notmatch $erwAug) {
        Write-Host "ABBRUCH: die Augmentierungsstaerke ist nicht die alte."
        Write-Host "Phase 8 laeuft mit 0.03 / 0.93-1.07. Steht dort die"
        Write-Host "Phase-6-Staerke, bewegen sich zwei Dinge gleichzeitig."
        exit 1
    }
    # 4. Der Beleg aus den DATEN, dass das ganze Bild gemeint war.
    if ($text -notmatch $erwAbdeckung) {
        Write-Host "ABBRUCH: die Kastenabdeckung liegt nicht bei 0.11x."
        Write-Host "Bei rund 0.18 haette der Lauf auf dem ZUSCHNITT gerechnet."
        Write-Host "Die Sollwerte je Fold stehen in der Vorfestlegung."
        exit 1
    }

    # ---- Der physikalische Gegentest -------------------------------------
    # Die Epochenzeit kann keine falsche Befehlszeile faelschen. Vierfache
    # Bildflaeche kostet Zeit, und wenn sie es nicht tut, ist die Kantenlaenge
    # irgendwo unterwegs verlorengegangen. Der Waechter Nummer 1 sollte das
    # schon gefangen haben; dies ist die zweite, unabhaengige Messung.
    $ep = [regex]::Matches($text, 'epoch \d+/\d+.*?\[(\d+)s,')
    if ($ep.Count -eq 0) {
        Write-Host "  Hinweis: keine Epochenzeit im Protokoll gefunden, der"
        Write-Host "  Zeitwaechter faellt fuer diesen Fold aus."
    } else {
        $sek = [int]$ep[0].Groups[1].Value
        Write-Host ("  erste Epoche {0} s (erwartet rund {1} s, 224 px waren 262 s)" `
                    -f $sek, $ERWARTET_EPOCHE_S)
        if ($sek -lt $MIN_EPOCHE_S) {
            Write-Host "ABBRUCH: die Epoche lief in $sek s, vorfestgelegte"
            Write-Host "Untergrenze ist $MIN_EPOCHE_S s. Vierfache Bildflaeche"
            Write-Host "kann nicht so schnell sein. --size ist nicht dort"
            Write-Host "angekommen, wo es wirkt."
            exit 1
        }
    }

    if (-not (Test-Path "$dir\head_f${f}_s0.npz")) {
        Write-Host "ABBRUCH: das Kopffeld wurde nicht geschrieben."; exit 1
    }
    Write-Host ("  fold ${f} fertig in $((Get-Date) - $foldStart)")
    Write-Host ("  Kontrollen: Adapter $DmlIndex, Entkopplung 0.500, Kopf da, " +
                "Eingang ${SIZE}px gemessen, ganzes Bild, alte Augmentierung, " +
                "Abdeckung 0.11x, Epochenzeit plausibel")
}

Write-Host "`nGesamtdauer $((Get-Date) - $start)"

$fertigeFolds = @(0, 1, 2, 3, 4) | Where-Object { Test-FoldFertig $_ }
Write-Host "fertige Folds in ${dir}: $($fertigeFolds -join ', ')"

if ($fertigeFolds.Count -lt 5) {
    Write-Host "`nNoch nicht alle fuenf Folds da. Das Urteil braucht alle fuenf."
    Write-Host "Weiter mit:  .\run_phase8.ps1"
    Write-Host ""
    Write-Host "Was jetzt angesehen werden darf, und was nicht:"
    Write-Host "  ANSEHEN: die Waechter oben, die Epochenzeit, best_epoch, und"
    Write-Host "  ob A eingebrochen ist (Abbruchgrund 4: mehr als 0,02)."
    Write-Host "  NICHT ANSEHEN als Entscheidungsgrundlage: dC auf einem Fold."
    Write-Host "  Die Foldstreuung auf dC ist 0,0245. Ein einzelner Fold wuerde"
    Write-Host "  den Lauf etwa in der Haelfte der Faelle allein durch Rauschen"
    Write-Host "  abbrechen. Das steht so in der Vorfestlegung."
} else {
    Write-Host "`nAlle fuenf Folds fertig. Das Urteil:"
    Write-Host "  venv\Scripts\python.exe rsna\befunde\rsna_phase8_auswertung.py"
}
