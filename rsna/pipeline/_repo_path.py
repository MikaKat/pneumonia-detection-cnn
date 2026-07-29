"""Legt Repo-Wurzel und beide RSNA-Ordner auf den Importpfad.

Python legt beim direkten Aufruf nur den Ordner des SKRIPTS auf `sys.path`.
`rsna/befunde/rsna_cam_lung_check.py` findet damit `rsna_train` nicht, das eine
Ebene daneben in `rsna/pipeline/` liegt, und `rsna_make_masks.py` findet
`segmentation` nicht, das an der Wurzel liegt.

Wer diese Datei importiert, hat danach beides. Das ist der Grund, warum beim
Umsortieren KEINE einzige bestehende Import-Zeile geaendert werden musste:
`from rsna_train import ...` gilt unveraendert weiter.

Die Datei liegt bewusst zweimal im Repo, einmal je Unterordner. Ein gemeinsames
Exemplar liesse sich erst importieren, nachdem der Pfad gesetzt ist, und genau
das ist ihre Aufgabe. Dieselbe Loesung steht in `tests/_repo_path.py`.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (_ROOT, _ROOT / "rsna" / "pipeline", _ROOT / "rsna" / "befunde"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
