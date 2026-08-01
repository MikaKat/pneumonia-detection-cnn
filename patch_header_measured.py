"""Write the MEASURED five-fold outcome next to the predictions that were
made before the runs, in rsna/pipeline/rsna_train.py.

The predictions stay untouched on purpose. A pre-registration that gets
quietly edited after the fact is worth nothing; the point of this patch is
that the reader can see the prediction, the measurement, and the gap between
them. Two of the three predictions held, one did not.

Idempotent: run it twice and the second run reports "already applied" and
changes nothing.

    python patch_header_measured.py
"""

from pathlib import Path
import sys

TARGET = Path("rsna") / "pipeline" / "rsna_train.py"
MARKER = "MEASURED afterwards"

# (anchor, replacement). Anchors are copied verbatim from the file.
PATCHES = [
    (
        "  * PRIMARY: `AUC(model score -> ViewPosition)` must FALL. Baseline\n"
        "    0.8166 +- 0.0098 over five folds.\n",

        "  * PRIMARY: `AUC(model score -> ViewPosition)` must FALL. Baseline\n"
        "    0.8166 +- 0.0098 over five folds.\n"
        "    MEASURED afterwards at strength 0.5, paired within fold:\n"
        "    -0.0334 +- 0.0086, t = -8.72 over five folds, every fold\n"
        "    negative. The endpoint held.\n",
    ),
    (
        "  * SECONDARY: the STRATIFIED AUC must NOT fall. Baseline "
        "0.8449 +- 0.0147.\n",

        "  * SECONDARY: the STRATIFIED AUC must NOT fall. Baseline "
        "0.8449 +- 0.0147.\n"
        "    MEASURED afterwards at strength 0.5: -0.0144 +- 0.0086, paired\n"
        "    t = -3.72, every fold negative. The fall is real. It stays\n"
        "    0.0006 inside the tolerance of 0.015 that `rsna_crop_compare.py`\n"
        "    checks, so the approving verdict line of that script rests on a\n"
        "    margin far smaller than the effect it is judging. Report the two\n"
        "    numbers, never that verdict on its own.\n",
    ),
    (
        "    stand here before the run, because afterwards it reads like a "
        "step\n"
        "    backwards.\n",

        "    stand here before the run, because afterwards it reads like a "
        "step\n"
        "    backwards.\n"
        "    MEASURED afterwards at strength 0.5: the raw AUC falls by\n"
        "    0.0151 +- 0.0061, a third of the predicted 0.044. The direction\n"
        "    was right, the size was not, and the prediction stays in this\n"
        "    text because it was made before the run. Why the gap is open:\n"
        "    the two paths to the label are not additive, so what the model\n"
        "    loses on the projection channel it partly recovers from the\n"
        "    image. This patch does not test that reading.\n",
    ),
    (
        '        print("    The RAW AUC is expected to fall by about 0.044 '
        'as well. "\n'
        '              "That is the success signature,")\n'
        '        print("    not a regression. See the module header.")\n',

        '        print("    The RAW AUC is expected to fall as well. That is "\n'
        '              "the success signature, not a regression.")\n'
        '        print("    Predicted before the runs: about 0.044. Measured "\n'
        '              "over five folds at strength 0.5: 0.0151 +- 0.0061.")\n'
        '        print("    See the module header.")\n',
    ),
]


def main() -> int:
    if not TARGET.exists():
        print(f"NOT FOUND: {TARGET}")
        print("Run this from the repository root.")
        return 2

    text = TARGET.read_text(encoding="utf-8")

    if MARKER in text:
        print("already applied, nothing changed")
        return 0

    missing = [i for i, (a, _) in enumerate(PATCHES, 1) if text.count(a) != 1]
    if missing:
        # Refuse a half patch. A file that is only partly rewritten is worse
        # than one that was never touched.
        print("ABORT: anchor not found exactly once, patch numbers "
              f"{missing}.")
        print("The file has moved on since this patch was written. "
              "Nothing changed.")
        return 3

    for anchor, replacement in PATCHES:
        text = text.replace(anchor, replacement, 1)

    backup = TARGET.with_suffix(".py.bak")
    if not backup.exists():
        backup.write_text(TARGET.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"backup written: {backup}")
    TARGET.write_text(text, encoding="utf-8")
    print(f"patched: {TARGET}  ({len(PATCHES)} places)")
    print("check with:  python rsna\\pipeline\\rsna_train.py --help")
    return 0


if __name__ == "__main__":
    sys.exit(main())
