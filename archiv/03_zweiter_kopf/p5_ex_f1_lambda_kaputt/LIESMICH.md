# exclude Fold 1, der Lauf mit dem kaputten Lambda

Gerechnet am 05.08.2026, archiviert am 06.08.2026.

In diesem Lauf stand lambda bei **75 052 408** statt bei rund 1. Ursache: der
allererste Stapel enthielt kein annotiertes Bild, bei `--head-negatives
exclude` war der Kopfverlust damit exakt null, und der Abfang bei 1e-8 machte
aus dem Verhaeltnis der beiden Verluste eine Zahl in der Groessenordnung 1e8.
Der Rumpf trainierte danach praktisch nur noch auf die Kaesten.

Der Lauf ist nicht kaputt im Sinne von unbrauchbar, er ist ein anderes
Experiment als die vier anderen Folds. Deshalb liegt er hier statt geloescht zu
sein: er ist der Beleg fuer den Befund, und er ist der einzige Lauf der Phase 5
mit dem niedrigsten C von allen fuenfzehn (0,6864 gegen 0,7174 bis 0,7725).

Ersetzt durch einen Neulauf mit reparierter Lambda-Messung, siehe
`run_p5_fold1_neu.ps1` und `erklaerungen/17_lambda_reparatur.md`. Die zugehoerige
Zeile in `results_rsna.csv` bleibt dort stehen; die Auswertung nimmt die letzte
Zeile je Arm und Fold und meldet den Wiederholungslauf ausdruecklich.

Enthalten: die Vorhersagedateien, das Kopffeld, die Lernkurve, das Gewicht und
das Trainingslog des alten Laufs.
