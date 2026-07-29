"""Put the repository root on the import path.

The test suites live in `tests/` but import the pipeline modules, which sit one
level up. Python puts the *script's own* directory on `sys.path`, not the
working directory, so `python tests/test_rsna_masks.py` would not find
`rsna_make_masks` without this.

Importing this module has the side effect of fixing the path, so each suite
imports it before anything from the project. Keeping it in one file means the
rule is stated once rather than copied into six headers.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
