<#
  Phase 5, der zweite Kopf. Drei Arme, alle auf Adapter 1, alle mit voller
  Entkopplung.

  WARUM DREI UND NICHT ZWEI
  -------------------------
  Der vorhandene Bezugsarm (predictions_rsna_dml1\) lief OHNE --balance-view.
  Phase 5 laeuft MIT, weil das ausgelieferte Modell es auch tun wird. Gepaart
  heisst: nur eine Sache darf sich unterscheiden. Also braucht Phase 5 einen
  eigenen Bezugsarm, in dem alles so ist wie in den Kopf-Armen, nur ohne Kopf.

  Der alte Bezugsarm bleibt gueltig fuer das, wofuer er gerechnet wurde, die
  Gleichwertigkeit der beiden Chips. Als Vergleichspartner fuer Phase 5 ist er
  es nicht.

  DIE ARME
  --------
    ref   ohne Kopf                        predictions_p5_ref\        --tag _p5ref
    ex    Kopf, Negative ausgeschlossen    predictions_final_model\    --tag _p5head_ex
    em    Kopf, Negative als leeres Feld   predictions_p5_head_em\    --tag _p5head_em

  Eigener --pred-dir und eigener --tag je Arm. Ohne das ueberschreiben sich die
  Gewichte gegenseitig, und genau das ist in diesem Projekt schon einmal
  unbemerkt passiert.

  REIHENFOLGE UND DAUER
  ---------------------
  Rund 3 h 16 je Arm auf der RX 5500 XT, zusammen also knapp zehn Stunden. Das
  Skript ueberspringt fertige Folds, es darf also jederzeit abgebrochen und
  neu gestartet werden.

  ZUERST DER RAUCHTEST
  --------------------
      powershell -ExecutionPolicy Bypass -File .\run_phase5.ps1 -Rauchtest

  Der trainiert Fold 0 drei Epochen ohne Grad-CAM, in BEIDEN Kopfarmen, rund
  eine halbe Stunde, und prueft danach je Arm, ob das Kopffeld ueber dem
  LAGEPRIORE liegt. Faellt der Test, ist die Kastenfalle zugeschnappt oder der
  Kopf sagt ueberall Null, und die zehn Stunden waeren verloren gewesen.

  Beide Arme, weil `em` der Arm ist, der still degenerieren kann: dort bekommen
  rund drei Viertel der Bilder ein leeres Zielfeld, und ein Kopf, der ueberall
  Null sagt, hat darauf einen hervorragenden Verlust. Im langen Lauf steht `em`
  an dritter Stelle, waere also erst nach sechseinhalb Stunden zu sehen.

  Nur einer, wenn es schnell gehen muss:
      powershell -ExecutionPolicy Bypass -File .\run_phase5.ps1 -Rauchtest -Arme ex

  DANN DER LANGE LAUF
  -------------------
      powershell -ExecutionPolicy Bypass -File .\run_phase5.ps1

  Einzelne Arme:
      powershell -ExecutionPolicy Bypass -File .\run_phase5.ps1 -Arme ref
      powershell -ExecutionPolicy Bypass -File .\run_phase5.ps1 -Arme ex,em

  VORFESTLEGUNG, festgelegt am 04.08. VOR diesem Lauf
  ---------------------------------------------------
  Primaerer Endpunkt ist A, die geschichtete AUC. Die Regel, in derselben Form
  wie beim Bezugsarm-Vergleich und bei Phase 8:

    NICHTUNTERLEGENHEIT, Marge 0,01 geschichtete AUC, gepaart je Fold,
    90-Prozent-Intervall, TOST.

  Faellt A um weniger als 0,01, ist der Kopf gesetzt. Faellt er um mehr, ist er
  es nicht. Die 0,01 sind nicht gewaehlt, sondern der Mindestunterschied, auf
  den dieses Projekt ueberhaupt reagiert; zum Vergleich: die volle Entkopplung
  kostete rund 0,018.

  REIHENFOLGE DER AUSWERTUNG, und die ist Teil der Vorfestlegung:
    1. zuerst `exclude` gegen `empty`, entschieden am Vorsprung ueber dem
       Lagepriore (B), NICHT an A und nicht am Kopfverlust;
    2. danach A fuer den GEWINNER gegen predictions_p5_ref. Nur dieser eine
       Vergleich ist bestaetigend;
    3. A fuer den Verlierer wird berichtet, aber ausdruecklich als erkundend.
  Sonst waere es Aussuchen: zwei Vergleiche rechnen und den besseren melden.
#>

param(
    [int[]]$Folds = @(0, 1, 2, 3, 4),
    [string[]]$Arme = @("ref", "ex", "em"),
    [int]$DmlIndex = 1,
    [int]$Epochen = 8,
    [int]$CamN = 300,
    [switch]$Rauchtest
)

# KEIN $ErrorActionPreference = "Stop", und das ist kein Versehen.
#
# Unter "Stop" behandelt PowerShell 5.1 JEDE Zeile, die ein externes Programm
# nach stderr schreibt, als abbrechenden Fehler (NativeCommandError). Der
# DirectML-Treiber schreibt beim ersten Stapel eine harmlose Warnung dorthin:
#
#   UserWarning: The operator 'aten::log_sigmoid_forward' is not currently
#   supported on the DML backend and will fall back to run on the CPU.
#
# Sie stammt aus der KLASSIFIKATIONS-Verlustfunktion, steht wortgleich schon im
# Bezugsarm-Log vom 04.08. und kostet dort nichts: die Epochen lagen bei 257 bis
# 259 Sekunden. Mit "Stop" starb der Lauf trotzdem nach zwei Minuten.
#
# `run_bezugsarm_dml1.ps1` hat die Zeile nie gehabt, deshalb lief er durch.
# Gebraucht wird sie auch nicht: nach jedem Aufruf wird $LASTEXITCODE geprueft,
# und jede eigene Bedingung endet mit einem ausdruecklichen `exit 1`.
$py = ".\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
New-Item -ItemType Directory -Force -Path logs | Out-Null

# Ein Arm ist: Anzeigename, Zusatzargumente, Zielordner, Tag.
$Spec = @{
    ref = @{ name = "Bezugsarm ohne Kopf"
             extra = @()
             dir   = "predictions_p5_ref";     tag = "_p5ref" }
    ex  = @{ name = "Kopf, Negative ausgeschlossen"
             extra = @("--head", "--head-negatives", "exclude")
             dir   = "predictions_final_model"; tag = "_p5head_ex" }
    em  = @{ name = "Kopf, Negative als leeres Feld"
             extra = @("--head", "--head-negatives", "empty")
             dir   = "predictions_p5_head_em"; tag = "_p5head_em" }
}

if ($Rauchtest) {
    $Folds   = @(0)
    $Epochen = 3
    $CamN    = 0
    # BEIDE Kopfarme, nicht nur `ex`, und das ist der Punkt.
    #
    # Die Kastenfalle faengt ein Rauchtest fuer beide, die ist gemeinsam. Der
    # zweite Verdaechtige nicht: "der Kopf sagt ueberall Null" droht in `empty`
    # staerker, weil dort rund drei Viertel aller Bilder ein LEERES Zielfeld
    # bekommen und ein flacher Kopf darauf einen hervorragenden Verlust hat.
    # `em` laeuft im langen Lauf als dritter Arm, wuerde also erst nach rund
    # sechseinhalb Stunden zum ersten Mal ausgefuehrt. Eine Viertelstunde hier
    # schuetzt dort gut drei.
    #
    # Ein ausdrueckliches -Arme wird respektiert, `ref` aber herausgeworfen:
    # der Rauchtest prueft ein Kopffeld, und der Bezugsarm hat keines.
    if (-not $PSBoundParameters.ContainsKey('Arme')) { $Arme = @("ex", "em") }
    $Arme = @($Arme | Where-Object { $_ -ne "ref" })
    if ($Arme.Count -eq 0) {
        Write-Host "Der Rauchtest braucht einen Kopfarm, ex oder em."; exit 1
    }
    # Eigener Ordner UND eigener Tag je Arm, aus demselben Grund wie im langen
    # Lauf: zwei Arme, die sich denselben Tag teilen, ueberschreiben einander
    # die Gewichte, und genau das ist in diesem Projekt schon einmal unbemerkt
    # passiert.
    foreach ($a in $Arme) {
        $Spec[$a].dir = "predictions_p5_rauchtest_$a"
        $Spec[$a].tag = "_p5rauchtest_$a"
    }
}

$ergebnis = if ($Rauchtest) { "results_rsna_rauchtest.csv" } else { "results_rsna.csv" }
$start = Get-Date

Write-Host "`nPhase 5, zweiter Kopf   start $($start.ToString('yyyy-MM-dd HH:mm'))"
Write-Host "Adapter $DmlIndex, Arme: $($Arme -join ', '), Folds: $($Folds -join ', ')"
Write-Host "Epochen $Epochen, Grad-CAM $CamN, python $py"
if (-not $Rauchtest) {
    $h = 3.3 * $Arme.Count * ($Folds.Count / 5.0)
    Write-Host ("Geschaetzt {0:N1} Stunden, also fertig gegen {1}." -f $h,
                (Get-Date).AddHours($h).ToString('dd.MM. HH:mm'))
}

# ---- Wachhunde -----------------------------------------------------------
# Sekunden, keine GPU-Rechnung. Der erste prueft, dass --dml-index ueberhaupt
# ankommt; faellt er, ginge der Lauf still auf der APU zu Ende und das faellt
# sonst erst nach Stunden auf. Der zweite prueft die Kastenfalle, den Grund,
# aus dem selbstgebaute Lokalisationskoepfe "einfach nicht lernen".
foreach ($t in @("tests\test_rsna_hardware.py", "tests\test_rsna_kopf.py")) {
    Write-Host "`n=== Wachhund: $t ==="
    & $py -u $t 2>&1 | ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath "logs\p5_$([IO.Path]::GetFileNameWithoutExtension($t)).log" |
        Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Pruefungen nicht bestanden. Abbruch."; exit 1
    }
}

foreach ($arm in $Arme) {
    $s = $Spec[$arm]
    Write-Host "`n############################################################"
    Write-Host "# ARM $arm : $($s.name)   ->  $($s.dir)\  (Tag $($s.tag))"
    Write-Host "############################################################"

    foreach ($f in $Folds) {
        $fertig = "$($s.dir)\rsna_f${f}_s0.csv"
        if (Test-Path $fertig) {
            Write-Host "fold ${f}: schon fertig, uebersprungen"; continue
        }

        $log = "logs\p5_${arm}_f${f}.log"
        Write-Host "=== $arm fold ${f}  start $(Get-Date -Format 'HH:mm')  -> $log ==="

        # Die Argumentliste wird als EIN Feld gebaut und gesplattet. Ein leeres
        # `$s.extra` mitten in einer Aufrufzeile kann PowerShell 5.1 als leere
        # Zeichenkette weiterreichen, und argparse bricht dann mit einer
        # Meldung ab, die nach einem Codefehler aussieht und keiner ist.
        $cmd = @("-u", "rsna\pipeline\rsna_train.py",
                 "--fold", $f, "--epochs", $Epochen, "--batch", 16,
                 "--workers", 0, "--cam-n", $CamN,
                 "--dml-index", $DmlIndex, "--balance-view") +
               $s.extra +
               @("--pred-dir", $s.dir, "--tag", $s.tag, "--out", $ergebnis)

        # Der Aufruf selbst gehoert IN das Log, nicht nur auf den Bildschirm.
        # Dieselbe Lehre wie bei `dml_index` und beim Tag: eine Herkunft, die
        # nur in einer Kommandozeile steht, ist eine Herkunft, die verloren
        # geht. Wer das Log Monate spaeter liest, sieht in Zeile 1, welcher Arm
        # das war, mit welchen Schaltern und in welchen Ordner.
        # (Ueberschreibt das Log; der Lauf darunter haengt mit -Append an.)
        "$py $($cmd -join ' ')" | Tee-Object -FilePath $log | Out-Host

        # ForEach-Object macht aus stderr-Zeilen Text, sonst zeigt PowerShell 5.1
        # die harmlose DirectML-Warnung als roten Fehlerblock. -u haelt das Log
        # waehrend des Laufs lesbar.
        & $py @cmd 2>&1 |
            ForEach-Object { $_.ToString() } |
            Tee-Object -FilePath $log -Append | Out-Host

        if ($LASTEXITCODE -ne 0) {
            Write-Host "$arm fold ${f} FEHLGESCHLAGEN, Exitcode $LASTEXITCODE. Abbruch. Log: $log"
            exit 1
        }

        # ---- Rauchtests auf die WIRKUNG, nicht auf die Ausgabe ----------
        # Der teuerste Fehler dieses Projekts war ein Lauf, der etwas anderes
        # tat, als er ankuendigte. Deshalb wird jedes Log geprueft.
        $text = Get-Content $log -Raw

        if ($text -notmatch "Hardware: directml:$DmlIndex") {
            Write-Host "ABBRUCH: im Log steht nicht 'Hardware: directml:$DmlIndex'."
            Write-Host "Der Lauf ging auf den falschen Chip. Alles ab hier waere wertlos."
            exit 1
        }
        if ($text -notmatch 'balance-view at strength') {
            Write-Host "ABBRUCH: kein Umgewichtungsblock im Log. Dieser Arm MUSS"
            Write-Host "umgewichten, sonst ist er nicht der Partner der anderen."
            exit 1
        }
        if ($text -notmatch 'in the stream: 0\.500') {
            Write-Host "ABBRUCH: die Entkopplung ist nicht vollstaendig angekommen."
            Write-Host "Im Strom muss AUC(ViewPosition -> label) auf 0.500 stehen."
            exit 1
        }
        if ($arm -eq "ref") {
            if ($text -match '--head:') {
                Write-Host "ABBRUCH: der Bezugsarm hat einen Kopf trainiert."; exit 1
            }
        } else {
            if ($text -notmatch '--head: 14 x 14 field') {
                Write-Host "ABBRUCH: kein 14 x 14 Kopf im Log."; exit 1
            }
            if ($text -notmatch 'lambda measured on batch') {
                Write-Host "ABBRUCH: lambda wurde nicht gemessen."; exit 1
            }
            # Die Stapelnummer steht bewusst NICHT in der Bedingung. Ein
            # Stapel ohne annotiertes Bild ist erlaubt, seit die Messung auf
            # den naechsten wartet; was verboten bleibt, ist ein lambda
            # ausserhalb der Groessenordnung, und dagegen bricht bereits
            # rsna_train.py selbst ab.
            if ($text -match 'measurement moved to batch') {
                Write-Host "  Hinweis: der erste Stapel hatte keinen Kasten, lambda kam vom naechsten."
            }
            if (-not (Test-Path "$($s.dir)\head_f${f}_s0.npz")) {
                Write-Host "ABBRUCH: das Kopffeld wurde nicht geschrieben."; exit 1
            }
        }
        Write-Host "  Kontrollen: Adapter $DmlIndex, Entkopplung 0.500, Arm wie angekuendigt"

        if (-not $Rauchtest -and $text -notmatch 'AUC stratified') {
            Write-Host "WARNUNG: keine geschichtete AUC im Log. Bitte $log ansehen."
        }
    }
}

Write-Host "`nGesamtdauer $((Get-Date) - $start)"

# ---- Der eigentliche Rauchtest: schlaegt der Kopf den Lagepriore? --------
if ($Rauchtest) {
    Write-Host "`n############################################################"
    Write-Host "# Kopffeld gegen den Lagepriore"
    Write-Host "############################################################"
    # Je Arm ein eigenes Urteil. Ein gemeinsames waere hier falsch: die beiden
    # unterscheiden sich gerade darin, WIE sie mit Bildern ohne Kasten umgehen,
    # und ein Arm, der durchfaellt, waehrend der andere haelt, ist selbst schon
    # ein Befund. Gesammelt wird der schlechteste Ausgang.
    $rc = 0
    $durchgefallen = @()
    foreach ($arm in $Arme) {
        # Erst in eine eigene Variable, dann uebergeben. In der Argumentzeile
        # eines externen Programms parst PowerShell 5.1 zusammengesetzte
        # Ausdruecke nicht zuverlaessig, und ein Ordnername, der als literales
        # "$Spec[ex].dir" ankaeme, wuerde hier still das Falsche pruefen.
        $rdir = $Spec[$arm].dir
        Write-Host "`n--- Arm $arm : $($Spec[$arm].name)   ->  $rdir\ ---"
        & $py rsna\befunde\rsna_kopf_auswertung.py rauchtest `
            --pred-dir $rdir --folds 0
        if ($LASTEXITCODE -ne 0) {
            $durchgefallen += $arm
            if ($LASTEXITCODE -gt $rc) { $rc = $LASTEXITCODE }
        }
    }
    Write-Host ""
    if ($rc -eq 0) {
        Write-Host "Rauchtest bestanden, Arme: $($Arme -join ', '). Der lange Lauf"
        Write-Host "ist startklar:"
        Write-Host "  powershell -ExecutionPolicy Bypass -File .\run_phase5.ps1"
        Write-Host "Die Ordner predictions_p5_rauchtest_* und"
        Write-Host "checkpoints\*_p5rauchtest_*.pth duerfen danach weg, ebenso"
        Write-Host "results_rsna_rauchtest.csv."
    } else {
        Write-Host "Rauchtest NICHT bestanden in: $($durchgefallen -join ', ')."
        Write-Host "Den langen Lauf nicht starten. Die drei Verdaechtigen stehen"
        Write-Host "in der Ausgabe oben."
        if ($durchgefallen.Count -lt $Arme.Count) {
            Write-Host ""
            Write-Host "ACHTUNG, und das ist die interessante Lage: der andere Arm"
            Write-Host "hat gehalten. Dann ist es NICHT die Kastenfalle, die traefe"
            Write-Host "beide gleich. Verdaechtig ist dann pos_weight je Kachel oder"
            Write-Host "ein Kopf, der auf den leeren Zielfeldern flach wird."
        }
    }
    exit $rc
}

# ---- Nach dem langen Lauf ------------------------------------------------
$fehlt = @()
foreach ($arm in $Arme) {
    foreach ($f in $Folds) {
        if (-not (Test-Path "$($Spec[$arm].dir)\rsna_f${f}_s0.csv")) {
            $fehlt += "$arm/f$f"
        }
    }
}
if ($fehlt.Count -gt 0) {
    Write-Host "`nNoch offen: $($fehlt -join ', ')"
    Write-Host "Skript einfach erneut starten, fertige Folds werden uebersprungen."
    exit 1
}

Write-Host "`nAlle Arme fertig. Naechster Schritt, ohne Rechenzeit:"
Write-Host "  $py rsna\befunde\rsna_kopf_auswertung.py rauchtest --pred-dir predictions_final_model --folds 0 1 2 3 4"
Write-Host "  $py rsna\befunde\rsna_kopf_auswertung.py rauchtest --pred-dir predictions_p5_head_em --folds 0 1 2 3 4"
Write-Host ""
Write-Host "Danach der eigentliche Vergleich auf Endpunkt A (geschichtete AUC),"
Write-Host "gepaart je Fold gegen predictions_p5_ref\. Die Vorfestlegung steht in"
Write-Host "erklaerungen\00_roadmap_v1.md, Phase 5: B ist hier KEIN Endpunkt."
