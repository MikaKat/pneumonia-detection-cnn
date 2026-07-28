# Plan für die nächste Session

Stand: 28.07.2026. Reihenfolge von Mika festgelegt. Zwei Punkte brauchen vorher
eine Entscheidung — sie sind unten mit **ENTSCHEIDUNG** markiert.

Ausgangslage: [`READMEforMe.md`](READMEforMe.md) ist die vollständige Arbeitsakte.
Kurzfassung des Stands:

- RSNA-Basislinie steht: geschichtete AUC **0,845 ± 0,015**, Abstand zur
  Header-Baseline **0,288 ± 0,011**, Grad-CAM lokalisiert Faktor 4,6.
- Segmentierung als Vorverarbeitung (pixelgenaue Maske) ist **verworfen**, alle
  drei Wirkwege einzeln widerlegt.
- Externe Validierung auf Kermany **geglückt**: leak-bereinigt 0,885 gegen
  intern 0,845. Kalibrierung überträgt sich nicht (NPV 0,500).
- Offen und unangetastet: **Holdout, 3812 Bilder, genau einmal auswerten.**

---

## 1. Crop-Idee umsetzen und testen

Zuschnitt auf das **quadratische** umschließende Rechteck der Lungenmaske plus
Rand. Nicht pixelgenau maskieren — das ist die verworfene Variante.

**Was schon feststeht** (aus `rsna_crop_geometry.py`):

- Zuschnitt-Rechteck belegt 0,63 des Bildes, linearer Zoom **1,27×**.
- Rahmung sagt die Projektion mit **AUC 0,745** vorher; davon ist das
  Seitenverhältnis (0,623) nicht entfernbar, weil es Anatomie ist.
  Entfernbar ist der Rest, rund +0,12.
- Zuschnitt-Parameter verraten die Klasse **nicht** eigenständig: geschichtet
  nach ViewPosition bleiben 0,541. Kein neuer Shortcut.
- **Quadratisch** zuschneiden, nicht die Seiten einzeln strecken — sonst wird
  der Projektionskanal als Verzerrung neu kodiert statt geschlossen.

**Endpunkte vor dem Start festlegen** (sonst wird hinterher die Zahl gesucht,
die am besten aussieht):

- **Primär: Modellscore → ViewPosition.** Jetzt 0,808. Fällt er, hat der Crop
  die Confounder-Abhängigkeit gesenkt. Das ist das erklärte Ziel.
- Sekundär: geschichtete AUC, **gepaart im selben Fold**.
- Kontrolle: Grad-CAM-Trefferquote gegen die Boxen (Boxkoordinaten müssen
  mittransformiert werden!).

**Voraussetzung, die noch fehlt:** Masken liegen nur für **4064** Bilder vor.
Für ein Training werden alle **22 872** Entwicklungsbilder gebraucht (Holdout
später separat). Grobe Schätzung 3–4 h auf CPU; `rsna_make_masks.py --device
directml` sollte deutlich schneller sein (der Checkpoint-Ladefehler ist behoben).

**Zuschnitt vorberechnen, nicht zur Laufzeit.** Analog zu `rsna_prepare.py` ein
`data/rsna/crop512/`. Zur Laufzeit zu croppen hieße, je Bild die Maske zu laden
— und unter DirectML läuft der DataLoader ohnehin mit `--workers 0`.

## 2. Code bereinigen und wegsortieren

**Archivieren, nicht löschen.** Die Ergebnisdateien *sind* die Evidenz für die
Mappe; ohne sie ist jede Zahl im README eine Behauptung.

- Behalten und ordnen: `results*.csv`, `predictions*/`, `qc/`,
  `diagnostics_results/`, `READMEforMe.md`, Checkpoints.
- Vorschlag: `archiv/phase1_kermany/`, `archiv/phase2_masken/`,
  `archiv/phase3_rsna/` — je mit einer kurzen `WAS_IST_DAS.md`.
- Kermany-Skripte (`train_compare.py`, `data_masked.py`, `mask_leakage_check.py`
  …) sind abgeschlossene Phasen: ins Archiv, nicht in den Papierkorb.
- `PLAN_naechste_session.md` (diese Datei) am Ende auflösen.

**Git ist in schlechtem Zustand:** 3 Commits, große Mengen untracked. Vor dem
Aufräumen einmal sauber committen, sonst ist das Aufräumen nicht rückgängig
zu machen. `.gitignore` für `data/`, `venv/`, `*.pth` prüfen — Checkpoints und
Rohdaten gehören nicht ins Repo.

## 3. Langes Training auf Crop

**ENTSCHEIDUNG NÖTIG — hier steckt ein Denkfehler.**

Ein 24-h-Crop-Lauf gegen die vorhandene 8-Epochen-Basislinie verglichen misst
**Epochen, nicht Crop**. Die etablierte Regel des Projekts lautet: gepaart, bei
gleichem Budget. Sonst bedeutet die Differenz nichts.

Zwei saubere Wege:

- **A — getrennt.** Erst die Entscheidungsfrage bei *gleichem* Budget
  (8 Epochen, gepaart je Fold, ~2,3 h je Lauf). Dann die Gewinnervariante lang
  trainieren als finales Modell. Kosten: 2,3 h + 24 h.
- **B — beides lang.** Crop und Basislinie je 24 h. Sauber, aber ~48 h.

**Empfehlung: A.** Die lange Laufzeit beantwortet die Crop-Frage nicht, sie
liefert das finale Modell.

**Für die Grafiken fehlt Protokollierung.** `rsna_train.py` schreibt derzeit nur
eine Ergebniszeile am Ende. Gebraucht wird eine `history_f{fold}.csv` je Epoche:
Train-Loss, Val-Loss, Val-AUC, geschichtete AUC, Lernrate, Zeit. Ohne die gibt
es keine Lernkurve zum Zeigen.

## 4. Webapp verbessern

`webapp/` stammt aus der Kermany-Phase und bedient das alte Modell.

**Der wichtigste Punkt ist kein technischer.** Die externe Validierung hat
gezeigt: bei übertragener Schwelle ist der **NPV 0,500** — von den als negativ
eingestuften Fällen hat die Hälfte tatsächlich Pneumonie. Eine Oberfläche, die
ein binäres „Pneumonie / keine Pneumonie" ausgibt, wäre damit irreführend.

- Wahrscheinlichkeit plus Unsicherheit zeigen, nicht ein Label.
- Grad-CAM mitliefern — inklusive des ehrlichen Hinweises, dass die Karte grob
  richtig zeigt (Faktor 4,6 beim Maximum), aber diffus ist (Faktor 1,6 bei der
  Masse).
- Deutlicher Hinweis: Forschungsdemonstrator, keine diagnostische Verwendung.
- Für einen Radiologen als Leser ist genau diese Zurückhaltung das Argument.

## 5. Öffentliches README

Das ist der eigentliche Bewerbungstext. Der Stoff liegt vollständig vor.

Tragende Punkte, in dieser Reihenfolge:

1. Jede Zahl steht neben ihrer Nullhypothese (Header-Baseline 0,557,
   Abstand 0,288 statt nackter 0,880).
2. Metadaten-Leaks gefunden und beziffert statt weggelassen (Kermany 0,915).
3. Confounder geschichtet statt versteckt (ViewPosition, +0,044).
4. **Negativergebnisse mit Beleg**: Segmentierung trägt hier nichts, drei Wege
   einzeln widerlegt, 11,5 h begründet gespart.
5. **Externe Validierung**: 0,885 gegen 0,845 intern.
6. **Kalibrierung überträgt sich nicht** (NPV 0,500) — der Unterschied zwischen
   „AUC 0,92" und „einsatzfähig".

Punkt 4 und 6 sind das Unterscheidungsmerkmal. Ein Portfolio, das eine gute
Zahl zeigt, gibt es oft; eines, das zeigt, wie die Zahl gegen die eigenen
Zweifel verteidigt wurde, selten.
