# Archiv: die Kermany-Phase

Hier liegt die erste Hälfte des Projekts — abgeschlossen, aber **nicht wertlos**.
Die Ergebnisdateien sind die Evidenz für die Zahlen im README. Ohne sie ist jede
davon eine Behauptung.

Archiviert am 28.07.2026, weil die Arbeit auf RSNA weitergeht und das
Wurzelverzeichnis sonst 35 Skripte aus drei Phasen nebeneinander enthielte.

## Was hier passiert ist, in einem Absatz

Ein ResNet-18 auf dem Kermany-Datensatz (Kinderthorax, Guangzhou) erreichte
schnell eine hohe AUC — und Grad-CAM zeigte nie auf ein Infiltrat. Die Suche nach
dem Grund hat den eigentlichen Wert dieser Phase ausgemacht: das Modell las
**globale Textur und Kontrast** statt Radiologie. Dahinter steckte ein
**Metadaten-Leak** — allein die JPEG-Abmessungen trennen die Klassen mit
AUC 0,915, ohne einen einzigen Bildpunkt. Per-Bild-Standardisierung und
Rebalancing haben den Kurzschluss geschlossen, aber der Datensatz blieb an seiner
Decke. Die Konsequenz war der Wechsel auf RSNA, wo die Störgrößen im
DICOM-Header stehen und gemessen statt rekonstruiert werden können.

Die zweite Hälfte dieser Phase war der Versuch, per **Lungensegmentierung**
vorzuverarbeiten. Auch der ist gescheitert, und auch das mit Beleg: die
Maskenfläche allein sagte die Klasse mit AUC 0,255 vorher — die Maske hat einen
neuen Shortcut gebaut statt einen alten zu entfernen. Wie weit das trägt, steht
in `READMEforMe.md`, Abschnitt „Schritt 9g".

## Aufbau

```
code/         die Skripte dieser Phase
ergebnisse/   CSVs, Vorhersagen, Diagnosebilder — die Belege
qc/           Merkmalsverteilungen und Schärfe-Analysen
```

### `code/`

| Datei | wofür |
| --- | --- |
| `train.py` | erstes Training, Kermany |
| `train_masked.py` | Training auf maskierten Bildern |
| `train_compare.py` | Variantenvergleich, gepaart je Fold |
| `evaluate.py` | Metriken |
| `gradcam.py` | Heatmaps |
| `diagnostics.py` | die Störungstests (Ecken, Zoom, Unschärfe, Kontrast) |
| `data_masked.py` | Dataset mit Maskierung |
| `lung_preprocess.py` | Maskenerzeugung für Kermany |
| `mask_leakage_check.py` | **fand den Maskenflächen-Leak (AUC 0,255)** |
| `metadata_leak_check.py` | **fand den Metadaten-Leak (AUC 0,915)** |
| `sharpness_leak_check.py` | Schärfe als Confounder, gematchte AUC 0,744 |
| `test_metrics.py` | Tests zu `train_compare.py` |

### `ergebnisse/`

`results.csv`, `results_compare.csv`, `bench.csv`, `smoke.csv`, `splits.json`,
`predictions/`, `diagnostics_results/` (Basislinie, nach Normalisierung, nach
Rebalancing, sowie die beiden Maskierungs-Varianten `phase2_v3` und `phase2_v4`).

## Wenn hier etwas erneut laufen soll

Die Skripte wurden aus dem **Wurzelverzeichnis** ausgeführt und importieren
`data.py` und `model/`. Beide sind absichtlich dort geblieben: der
`Dockerfile` der Webapp kopiert sie, und sie zu verschieben hätte den Build
zerlegt. Also aus dem Wurzelverzeichnis starten:

```powershell
$env:PYTHONPATH = "."
python archiv/kermany/code/diagnostics.py
```

Ebenfalls im Wurzelverzeichnis geblieben, mit Absicht:

- **`splits.py`** — `rsna_external_kermany.py` benutzt `parse_record` daraus.
  Die externe Validierung ist ein *laufendes* Ergebnis, keine abgeschlossene
  Phase. Nur die zugehörige `splits.json` liegt hier im Archiv, weil sie allein
  die Kermany-Aufteilung enthält.
- **`segmentation/`** — `rsna_make_masks.py` lädt `unet.py` und
  `mask_refine.py`. Das U-Net stammt aus dieser Phase, wird aber weiter benutzt.
- **`data.py`, `main.py`, `model/`, `checkpoints/best_model.pth`, `samples/`** —
  die Webapp und der Docker-Build hängen daran.
