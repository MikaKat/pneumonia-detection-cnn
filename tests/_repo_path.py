"""Put the repository root and both RSNA folders on the import path.

The test suites live in `tests/` but import the pipeline modules, which sit in
`rsna/pipeline/` and `rsna/befunde/`. Python puts the *script's own* directory
on `sys.path`, not the working directory, so `python tests/test_rsna_masks.py`
would not find `rsna_make_masks` without this.

Importing this module has the side effect of fixing the path, so each suite
imports it before anything from the project. Keeping it in one file means the
rule is stated once rather than copied into six headers.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "rsna" / "pipeline", _ROOT / "rsna" / "befunde", _ROOT / "serving"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
