"""
Sanity run on RSNA: one fold, one model, all controls.

A fold is one patient-grouped division of the data into a training part and a
reporting part, and this script runs exactly one of them. It writes a metrics
row to results_rsna.csv, the per-image predictions for the reporting split and
for the inner selection split, a per-epoch history CSV, the Grad-CAM table and
the weights of the selected epoch.

The aim is not the best value but a defensible first one, plus an answer to
four questions:

  1. Does the model beat the header-only baseline of 0.729? AUC is the
     probability that a random pneumonia case is ranked above a random
     non-case, the same quantity as a c-statistic, and 0.729 is what a
     classifier reaches that never sees an image. Anything below means the
     model learned less than the header alone carries.
  2. Does it beat that baseline within one projection? There it sits at 0.553
     (AP) and 0.559 (PA). This is the actual question. The overall AUC still
     contains the AP/PA effect, the stratified one does not.
  3. Does Grad-CAM point at the pathology? Grad-CAM turns the evidence the
     network used into a heatmap over the film. RSNA ships bounding boxes, so
     this is measurable instead of a matter of opinion. It was the starting
     question of the whole project.
  4. Does the model read the burnt-in markers? "PORTABLE" is printed on the AP
     films. The corner ablation, which blanks the four corners and repeats the
     evaluation, answers that.

How to read the output: auc goes against 0.729, auc_stratified against roughly
0.556, and cam_hit against cam_area_baseline, the fraction of image area the
boxes cover. A stratified AUC down at its baseline, or a hit rate no better
than the box area, means the image contributed nothing. A large drop under the
corner ablation means the marker was doing the work.

Carried over from phase 3, where it was learned the expensive way:
  * Checkpoint and threshold both come from an inner, patient-grouped
    selection split. The outer val is only reported, never optimised.
  * `auc_last` and `*_oracle` run along as an optimistic reference so the gap
    stays visible.
  * Every prediction goes to disk as CSV. On Kermany each follow-up question
    otherwise cost a full retraining run.

New against phase 3:
  * No caliper matching. The confounder here is binary and exactly known, so
    the metrics are stratified instead. That costs not a single image, while
    the matching on Kermany discarded two thirds of the data.
  * No `RandomResolution`. That jitter was built against the Kermany zoom
    confounder. Here all images are the same size and it would only add noise.

--balance-view: taking the incentive away instead of the pixels
---------------------------------------------------------------
Three attempts to remove the projection from the images all failed, and the
crop made it worse. Masking rewrites the channel into the lung silhouette,
where 0.692 of 0.714 survives. The crop rewrites it into the magnification
factor: the window side alone predicts AP against PA at 0.685. Per-image
normalisation rewrites it into the relative intensity of the lung, 0.721 rising
to 0.768. A deterministic transform can re-encode information, it cannot delete
it.

The reason the model reads the projection is not that the projection is
visible. It is that the projection is USEFUL: `ViewPosition -> Target` has AUC
0.706, so a model trained on label accuracy has every reason to encode it. The
evidence for the projection is spread over heart size, scapulae, diaphragm
position, framing and sharpness at once, which is why removing any single
carrier changes nothing. The incentive sits in one place only.

`--balance-view` removes the incentive. Every training image is drawn with a
weight that makes projection and label statistically independent in the
training stream, and the weight is the ratio a chi-square test is built on,
expected count under independence divided by observed count:

    w(v, y) = n_v * n_y / (N * n_vy)

On the development set that is AP-negative 1.26, AP-positive 0.59, PA-negative
0.85, PA-positive 2.42. Both MARGINALS stay exactly as they were, the overall
prevalence of 0.225 and the AP to PA ratio; only the association between them
is cut. That is deliberate, because it keeps `pos_weight` valid and the class
imbalance is not corrected twice.

The weights come from the FITTING SPLIT of the current fold alone, never from
the whole development set, or fold information leaks. Selection split and
reporting split are never reweighted, otherwise nothing stays comparable to the
baseline.

How to read the result, written down before the first run:

  * PRIMARY: `AUC(model score -> ViewPosition)` must FALL. Baseline
    0.8166 +- 0.0098 over five folds.
    MEASURED afterwards at strength 0.5, paired within fold:
    -0.0334 +- 0.0086, t = -8.72 over five folds, every fold
    negative. The endpoint held.
    At strength 1.0, measured on the same five folds:
    -0.0554 +- 0.0123, t = -10.05.
  * SECONDARY: the STRATIFIED AUC must NOT fall. Baseline 0.8449 +- 0.0147.
    MEASURED afterwards at strength 0.5: -0.0144 +- 0.0086, paired
    t = -3.72, every fold negative. The fall is real. It stays
    0.0006 inside the tolerance of 0.015 that `rsna_crop_compare.py`
    checks, so the approving verdict line of that script rests on a
    margin far smaller than the effect it is judging. Report the two
    numbers, never that verdict on its own.
    At strength 1.0: -0.0181 +- 0.0058, t = -7.03, which is
    OUTSIDE that tolerance. Paired against strength 0.5 the extra
    cost is -0.0037 +- 0.0042, t = -1.94, which cannot be told
    from noise, while the extra gain on the primary endpoint,
    -0.0219 +- 0.0156, t = -3.15, can. The tolerance is a line
    that was set, not one that was measured. Both doses lower the
    stratified AUC and in both the fall is real.
  * The RAW AUC IS EXPECTED TO FALL, and that is the success, not a
    regression. The 0.880 contains the +0.044 contributed by `ViewPosition`.
    Removing the channel costs roughly that much, so a raw AUC sinking towards
    0.845 is the predicted signature of the intervention working. This has to
    stand here before the run, because afterwards it reads like a step
    backwards.
    MEASURED afterwards at strength 0.5: the raw AUC falls by
    0.0151 +- 0.0061, a third of the predicted 0.044. The direction
    was right, the size was not, and the prediction stays in this
    text because it was made before the run. Why the gap is open:
    the two paths to the label are not additive, so what the model
    loses on the projection channel it partly recovers from the
    image. This patch does not test that reading.
    At strength 1.0, where the association is cut completely, the
    raw AUC still falls by only 0.0233 +- 0.0021. The gap to the
    predicted 0.044 is therefore not an artefact of a half dose.
  * Calibration shifts, so the per-stratum thresholds move. They are searched
    on the selection split, which the reweighting does not touch.

The price is 14 percent of the effective sample size, 19,698 of 22,872 by the
Kish measure. Drawing a PA-positive image 2.42 times does not create 2.42
patients. What it does create, since the draw passes through
`build_transforms` again, is 2.42 differently augmented views of it.

--head: der zweite Kopf, Phase 5
--------------------------------
Bis hierher beantwortet das Modell eine Frage, ob. Mit `--head` beantwortet es
zwei, ob und wo. Der Rumpf bleibt derselbe, daneben tritt ein zweiter Ausgang:
ein Feld von 14 mal 14 Zahlen, trainiert gegen die in dieses Raster
eingezeichneten annotierten Kaesten. Ausfuehrlich in
`erklaerungen/14_zweiter_kopf_bau.md`.

Fuenf Entwurfsentscheidungen, jede gemessen oder vorfestgelegt statt gewaehlt:

  1. RASTERWEITE 14 mal 14, entschieden von `rsna_kopfraster.py` auf 4123
     annotierten Bildern. Die Decke der Punkt-AUC trennt 7 und 14 nicht
     (0.9928 gegen 0.9989, weniger als die 0.01, die dieses Projekt aufloest);
     entschieden hat die Decke der IoU, 0.53 gegen 0.74, weil Phase 5b genau
     diese Zahl als Wettbewerbsmass berichtet. Dass das zweite Tor nachtraeglich
     dazukam, steht im Kopf von `rsna_kopfraster.py`.
  2. WO DER KOPF HAENGT: an `layer3`, nicht an `layer4`. Bei 224 Punkten liefert
     layer3 genau 14 mal 14, der Rumpf bleibt damit voellig unangetastet und der
     Klassifikationsweg ist Zeile fuer Zeile derselbe wie ohne Kopf. Die
     Alternative, layer4 mit veraenderter Schrittweite, haette den Rumpf
     geaendert und damit ZWEI Dinge zugleich, was den gepaarten Vergleich
     zerstoert. Ein `adaptive_avg_pool2d` auf 14 haelt das Raster fest, auch
     wenn spaeter mit 512 Punkten gerechnet wird. Genau das ist der Grund,
     warum Phase 5 vor Phase 8 kommt: das Lineal aendert sich nicht mehr mit
     dem Gemessenen.
  3. WEICHE ZIELE. Ziel einer Kachel ist der Anteil ihrer Flaeche, den ein
     Kasten bedeckt, nicht 0 oder 1. Der Preis des harten Ziels ist gemessen:
     bei 14 mal 14 erfindet es 14 Prozent Kastenflaeche dazu und verliert 16.
  4. VERLUSTGEWICHT, gerechnet statt geraten. Auf dem ersten Stapel MIT einem
     annotierten Bild werden beide Verluste gemessen und lambda so gesetzt,
     dass sie gleich gross starten. "Gleich gewichtet" ist damit eine Tatsache
     und keine Behauptung. Wert, Stapelnummer und ein Bereichswaechter stehen
     im Log und in results_rsna.csv. Der Zusatz "mit einem annotierten Bild"
     ist der ganze Punkt: bei `exclude` traegt ein Bild ohne Kasten nichts bei,
     und ein Stapel ohne jeden Kasten ergaebe eine Division durch Null.
  5. NEGATIVE, `--head-negatives`. Bilder ohne Pneumonie haben keinen Kasten.
     `exclude` nimmt sie aus dem Kopfverlust, `empty` laesst sie ein leeres Feld
     lernen. Beide Varianten werden gemessen, das ist selbst ein Befund.
     ENTSCHIEDEN WIRD AM VORSPRUNG UEBER DEM LAGEPRIORE, nicht am Kopfverlust:
     ein Kopf, der ueberall Null sagt, hat einen hervorragenden Verlust und ist
     wertlos. `pos_weight` je Kachel wird zur Variante passend aus dem
     Fit-Split gerechnet, nicht eingetragen.

DIE KASTENFALLE, der wichtigste Baupunkt
----------------------------------------
Sobald die Kaesten ins Training gehen, muss jede geometrische Augmentierung
AUCH auf die Kaesten wirken. Die Augmentierung dreht bis 7 Grad, verschiebt bis
3 Prozent und skaliert bis 7 Prozent. Ohne dieselbe Bewegung am Ziel lernt der
Kopf gegen systematisch verschobene Rechtecke, und die Aufsicht wird zu
Rauschen. Der Fehler ist unsichtbar: der Verlust faellt, das Skript laeuft
durch, nur die Karte bleibt diffus.

Deshalb gibt es `TrainTransform`. Sie zieht die Affinparameter EINMAL und wendet
sie auf Bild und Kastenmaske an. Helligkeit und Kontrast beruehren die Maske
nicht. `tests/test_rsna_kopf.py` prueft die Schwerpunkte beider nach der
Transformation gegeneinander.

`TrainTransform` laeuft in BEIDEN Armen, auch ohne Kopf. Das ist Absicht: liefe
der Arm ohne Kopf noch ueber das alte `build_transforms`, unterschieden sich die
Arme in der Augmentierungsmechanik UND im Kopf, also in zwei Dingen. Der Preis
ist, dass Laeufe ab hier nicht mehr bitgleich zu denen vor dem 04.08.2026 sind.
Das ist verkraftbar, weil der Bezugsarm fuer Phase 5 ohnehin neu gerechnet wird.

CLI:
  python rsna_train.py --fold 0 --epochs 8 --batch 16 --workers 0
  python rsna_train.py --fold 0 --epochs 8 --batch 16 --workers 0 --balance-view
  python rsna_train.py --fold 0 --balance-view --head --head-negatives exclude
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.models import ResNet18_Weights, resnet18

IMNET_MEAN, IMNET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BOX_SPACE = 1024          # boxes are given in the original DICOM grid
HEAD_GRID = 14            # measured, not chosen; see rsna_kopfraster.py
# Plausible range for a MEASURED lambda. Every run so far landed between 0.71
# and 1.27, so these bounds are three orders of magnitude wide on either side
# and can only catch a lambda that is wrong, not one that is unexpected. A
# value entered by hand through --head-lambda is not checked against them.
LAMBDA_MIN, LAMBDA_MAX = 1e-3, 1e3


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

class RsnaDataset(Dataset):
    """IDs instead of paths. The loader builds the path. See rsna_splits.py.

    With `boxes` given the item grows from (image, label) to
    (image, label, field, used). `field` is the localisation target on the head
    grid, `used` says whether this image counts towards the localisation loss.
    Without `boxes` the return value is exactly what it always was, so the
    single-headed arm is not touched by the extension.
    """

    def __init__(self, root: Path, ids: list[str], labels: dict[str, int], tf,
                 boxes: dict | None = None, grid: int = HEAD_GRID,
                 negatives: str = "exclude", size: int = 224):
        self.root, self.ids, self.labels, self.tf = root, ids, labels, tf
        self.boxes, self.grid, self.negatives, self.size = boxes, grid, negatives, size

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]
        img = Image.open(self.root / f"{pid}.png").convert("L")
        if self.boxes is None:
            return self.tf(img), float(self.labels[pid])

        b = self.boxes.get(pid)
        # The mask is drawn at the SAME edge length the image is resized to,
        # because the affine translation is measured in pixels of that grid. A
        # mask drawn at 224 and an image at 512 would move by different amounts
        # and the boxes would drift, which is the box trap in its quietest form.
        mask = box_mask_pil(b, self.size)
        x, m = self.tf(img, mask)
        field = tile_coverage(m, self.grid)
        # An image without a box carries an empty field either way; `used`
        # decides whether that empty field is a statement ("nothing here") or
        # simply no data. That is the whole `--head-negatives` question.
        used = 1.0 if b else (1.0 if self.negatives == "empty" else 0.0)
        return x, float(self.labels[pid]), field, used


def box_mask_pil(boxes, size: int, box_space: int = BOX_SPACE) -> Image.Image:
    """Bounding boxes as a black and white PIL image of edge length `size`.

    PIL and not a tensor, so the mask can go through exactly the same
    `TF.affine` call as the image. Same truncation as `cam_vs_boxes` and as
    `rsna_lokalisation.box_mask`, deliberately, so target and measurement see
    the same rectangle down to the pixel.
    """
    a = np.zeros((size, size), np.uint8)
    if boxes:
        s = size / box_space
        for bx, by, bw, bh in boxes:
            y0, y1 = max(int(by * s), 0), int((by + bh) * s)
            x0, x1 = max(int(bx * s), 0), int((bx + bw) * s)
            a[y0:y1, x0:x1] = 255
    return Image.fromarray(a, mode="L")


def tile_coverage(mask: Image.Image, grid: int) -> torch.Tensor:
    """The share of each tile covered by a box. The soft target, 1 x grid x grid.

    `adaptive_avg_pool2d` and not a reshape: at 224 pixels and 14 tiles the two
    are identical (16 by 16 blocks), at 512 pixels the edge length is not
    divisible and a reshape would raise. Pooling keeps the same code path for
    every image size, which is the point of a head grid that does not move.
    """
    a = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0)[None, None]
    return Fn.adaptive_avg_pool2d(a, grid)[0]


class MaskCorners:
    """Sets the four image corners to the median. Ablation against marker reading.

    The AP films carry "PORTABLE", side markers and arrows inside the image.
    That is a direct visual proxy for the acquisition type, and the acquisition
    type is the whole confounder. A crude statistical test (share of bright
    pixels in the corners) finds nothing, but a convolutional network reads
    text better than a brightness threshold, so this gets tried rather than
    argued about.

    The median and not black, on purpose: a black patch is a conspicuous
    feature in itself and would introduce a new edge.
    """

    def __init__(self, frac: float = 0.18):
        self.frac = frac

    def __call__(self, img: Image.Image) -> Image.Image:
        a = np.asarray(img).copy()
        k = int(min(a.shape[:2]) * self.frac)
        med = int(np.median(a))
        a[:k, :k] = med; a[:k, -k:] = med; a[-k:, :k] = med; a[-k:, -k:] = med
        return Image.fromarray(a)


def build_transforms(size: int, train: bool):
    # NO horizontal flipping: it creates situs inversus, mirrors the cardiac
    # silhouette and contradicts the side marker printed in the image.
    #
    # Training now goes through TrainTransform instead, see below. This
    # function stays because the evaluation path (train=False) is used
    # unchanged in four places and because `perturbed_transform` builds on the
    # same base list.
    base = [T.Grayscale(num_output_channels=3), T.ToTensor(),
            T.Normalize(IMNET_MEAN, IMNET_STD)]
    if not train:
        return T.Compose([T.Resize((size, size))] + base)
    return T.Compose([
        T.Resize((size, size)),
        T.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.93, 1.07)),
        T.ColorJitter(brightness=0.15, contrast=0.15),
    ] + base)


class TrainTransform:
    """The training augmentation, applied to image AND box mask at once.

    THE BOX TRAP. `T.RandomAffine` draws its parameters inside its own
    `__call__`. Calling it twice, once for the image and once for the mask,
    draws TWICE and moves the two by different amounts. The box would then sit
    somewhere the opacity is not, the head would be supervised against noise,
    and nothing in the output would say so: the loss falls, the script
    finishes, only the map stays diffuse. It is the most common reason a
    hand-built localisation head "just does not learn".

    The fix is to draw ONCE with `RandomAffine.get_params` and apply the same
    parameters to both. Geometry is shared, photometry is not: brightness and
    contrast do not move a rectangle, so the jitter only touches the image.

    Nearest neighbour for the mask keeps it binary. The image keeps the nearest
    neighbour interpolation it always had, so the pixels of the single-headed
    arm are produced the same way as before.

    The order of the random draws is IDENTICAL whether or not a mask is passed:
    `get_params` first, `ColorJitter` second, and the mask branch draws
    nothing. That is what allows the arm with the head and the arm without to
    be compared as a paired experiment rather than as two different recipes.

    STRENGTH IS AN ARGUMENT, and the defaults are the values every run up to
    phase 5 used. `TrainTransform(size)` therefore draws exactly what it drew
    before, down to the bit: `get_params` sees the same three arguments and the
    generator is not touched by the signature. Phase 6 raises translate and
    widens scale through `--aug-translate` and `--aug-scale`, so the two arms
    differ in those numbers and in nothing else.

    Rotation stays at 7 degrees and is an argument only for completeness. More
    is unphysiological on a chest film, and mirroring stays forbidden anywhere
    in this file: it produces situs inversus and contradicts the side marker
    printed into the image.

    THE PHOTOMETRIC STRENGTH IS AN ARGUMENT TOO, added 09.08.2026 for phase 9.
    Until then `brightness` and `contrast` sat hard wired at 0.15 inside this
    constructor while the geometric strengths were already arguments, which is
    the exact shape of `--balance-strength`: a number that a caller believes it
    controls and does not. The defaults are 0.15, so every run up to phase 8
    means the same thing it meant before.

    What the knob can reach was measured BEFORE the phase, not diagnosed after
    it: `rsna/befunde/rsna_photometrie_reichweite.py`. At 0.15 the jitter
    removes 4 percent of the global brightness cue and 22 percent of the global
    contrast cue, so the value every run so far used is, for this purpose,
    close to no jitter at all.
    """

    def __init__(self, size: int, translate: float = 0.03,
                 scale: tuple[float, float] = (0.93, 1.07),
                 degrees: float = 7.0, brightness: float = 0.15,
                 contrast: float = 0.15):
        self.size = size
        self.degrees = [-float(degrees), float(degrees)]
        self.translate = (float(translate), float(translate))
        self.scale = (float(scale[0]), float(scale[1]))
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.jitter = T.ColorJitter(brightness=self.brightness,
                                    contrast=self.contrast)
        self.finish = T.Compose([T.Grayscale(num_output_channels=3),
                                 T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)])

    def __call__(self, img: Image.Image, mask: Image.Image | None = None):
        img = TF.resize(img, [self.size, self.size])
        p = T.RandomAffine.get_params(self.degrees, self.translate, self.scale,
                                      None, [self.size, self.size])
        img = TF.affine(img, *p, interpolation=T.InterpolationMode.NEAREST, fill=0)
        x = self.finish(self.jitter(img))
        if mask is None:
            return x
        mask = TF.affine(mask, *p, interpolation=T.InterpolationMode.NEAREST, fill=0)
        return x, mask


# The probe image for the measurement below. 62 x 62 = 3844 = 31 * 124, so the
# 31 grey values 49 to 79 each appear 124 times and the mean is EXACTLY 64.0.
# That matters: `ImageEnhance.Contrast` blends towards the mean ROUNDED to an
# integer, and a mean of 63.5 would put a factor dependent wobble into the very
# statistic being read off. No value can clip either: at the largest factor
# this file allows, 79 * 2 = 158 and 64 + 2 * 15 = 94, both below 255.
_PROBE_LO, _PROBE_HI, _PROBE_N = 49, 79, 124
_PROBE_MEAN = 64.0
_PROBE_SD = float(np.std(np.repeat(np.arange(_PROBE_LO, _PROBE_HI + 1),
                                   _PROBE_N)))


def gemessene_jitter_staerke(tf: "TrainTransform", ziehungen: int = 512,
                             seed: int = 20260809) -> tuple[float, float]:
    """Die photometrische Staerke, GEMESSEN am Objekt, das der Loader benutzt.

    Dasselbe Muster wie `input_px` in Phase 8 und `head_lambda_measured` in
    Phase 5, und aus demselben Anlass: `--balance-strength` wurde einmal
    korrekt gerechnet und dann als Vorgabe weitergereicht, und nichts in der
    Ausgabe widersprach. `args.aug_brightness` ist eine Selbstauskunft. Was
    hier herauskommt, ist es nicht: es ist die Streuung der Faktoren, die
    `tf.jitter` wirklich zieht.

    Gelesen wird an einem Testbild mit bekanntem Mittelwert und bekannter
    Streuung. ColorJitter multipliziert den Mittelwert mit dem
    Helligkeitsfaktor, und die Streuung mit dem Produkt aus Helligkeits- und
    Kontrastfaktor; beide Faktoren lassen sich damit je Ziehung zurueckrechnen.
    Gleichverteilt auf [1-b, 1+b] hat eine Standardabweichung von b/sqrt(3),
    also ist b das sqrt(3)-fache der gemessenen Streuung.

    Das Abschneiden bei PIL verschiebt den Mittelwert um eine feste halbe
    Stufe. Fest heisst: es faellt aus einer Standardabweichung heraus und muss
    nicht korrigiert werden.

    GENAUIGKEIT, gemessen und nicht behauptet: die Helligkeit kommt auf 0,4
    Prozent genau heraus, der Kontrast liegt systematisch 6 bis 9 Prozent zu
    hoch, weil die Streuung eines auf ganze Grauwerte gerundeten Testbildes
    nach oben verzerrt ist. Das ist der Grund fuer die weite Schranke unten.
    Diese Zahl ist eine VERKABELUNGSPRUEFUNG und keine Kalibrierung: sie soll
    0,15 von 0,60 unterscheiden, und das ist ein Faktor vier.

    DER ZUFALLSSTROM WIRD UNVERAENDERT ZURUECKGEGEBEN. Sonst zoege dieses
    Messen dem Training Zufallszahlen weg, und ein Lauf waere nicht mehr
    derselbe wie ohne die Messung. Geprueft in tests/test_rsna_phase9.py.
    """
    if max(tf.brightness, tf.contrast) > 0.9:
        raise SystemExit(
            "ABORT: --aug-brightness/--aug-contrast above 0.9. torchvision "
            "clamps the lower end of the factor range at 0, so the draw would "
            "no longer be uniform on [1-b, 1+b] and this measurement would "
            "read a strength that is not the one in effect. At exactly 1.0 a "
            "drawn factor of 0 also makes the contrast unrecoverable, since "
            "an all black image has no spread left to read.")
    a = np.repeat(np.arange(_PROBE_LO, _PROBE_HI + 1, dtype=np.uint8), _PROBE_N)
    probe = Image.fromarray(a.reshape(62, 62), mode="L")
    zustand = torch.get_rng_state()
    try:
        fb = np.empty(ziehungen)
        fc = np.empty(ziehungen)
        torch.manual_seed(seed)
        for i in range(ziehungen):
            v = np.asarray(tf.jitter(probe), dtype=np.float64)
            fb[i] = v.mean() / _PROBE_MEAN
            fc[i] = v.std() / (fb[i] * _PROBE_SD)
    finally:
        torch.set_rng_state(zustand)
    w = float(np.sqrt(3.0))
    return float(fb.std(ddof=1) * w), float(fc.std(ddof=1) * w)


def view_balance_weights(y: np.ndarray, vp: np.ndarray,
                         strength: float = 1.0) -> np.ndarray:
    """Per-image sampling weights that decouple projection from label.

    Returns one weight per image. The weight depends only on which cell of the
    projection-by-label table the image falls into:

        w(v, y) = [ n_v * n_y / (N * n_vy) ] ** strength

    which is the expected count of that cell under independence divided by its
    observed count, the same ratio a chi-square test is built from. Cells that
    are over-represented relative to independence are drawn less often, cells
    that are under-represented more often.

    Two properties make this the mild version rather than the brutal one, and
    both are checked in `test_rsna_train.py`:

      * The MARGINALS are preserved exactly. Weighted, the projections keep
        their sizes and the overall prevalence stays at its original value.
        Only the association between the two is removed. That is why
        `pos_weight` in the loss can stay untouched: the class imbalance is
        still there and is still corrected exactly once.
      * The weights SUM TO N. So `num_samples=len(ids)` keeps epochs the same
        length and the runtime per epoch does not change.

    Balancing all four cells to equal size would also work but is the wrong
    trade here: it costs half the effective sample size (11,772 of 22,872 by
    the Kish measure) against 19,698 for this version, and it would additionally
    require setting `pos_weight` to 1.

    Any value of `vp` forms its own stratum, so "unknown" is handled without a
    special case. Very small strata get extreme weights and are reported by the
    caller rather than silently clipped: clipping would quietly leave part of
    the association in place while the printed intent says it is gone.

    STRENGTH is the dial, and it exists because the full correction is not
    free. On fold 0 it moved the primary endpoint by -0.070 and cost Grad-CAM
    localisation, and the whole loss sat in AP: hit rate 0.596 to 0.342 there,
    against 0.306 to 0.333 in PA. AP positives are drawn 0.59 times and PA
    positives 2.42, so the model trades localisation in the majority
    projection for localisation in the minority one.

    `strength` raises every weight to that power and renormalises, so 0 is the
    untouched baseline, 1 is full independence, and values in between trade the
    two effects off against each other. `residual_view_label_auc` reports what
    a given setting leaves standing, BEFORE an epoch is spent on it. Running
    0.5 next to 0 and 1 turns a single point into a dose-response curve, which
    is the stronger form of evidence: it shows the cost is a dial and not an
    accident.
    """
    y = np.asarray(y).astype(int)
    vp = np.asarray(vp).astype(str)
    n = len(y)
    if n == 0:
        return np.zeros(0, dtype=float)
    w = np.ones(n, dtype=float)
    for v in np.unique(vp):
        for c in np.unique(y):
            cell = (vp == v) & (y == c)
            n_vy = int(cell.sum())
            if n_vy == 0:
                continue
            w[cell] = (vp == v).sum() * (y == c).sum() / (n * n_vy)
    if strength != 1.0:
        w = w ** float(strength)
        # Renormalise, because w ** a no longer sums to n. Without this the
        # epoch would silently change length with the setting and any
        # comparison between settings would confound dose with training time.
        w *= n / w.sum()
    return w


def residual_view_label_auc(y: np.ndarray, vp: np.ndarray,
                            w: np.ndarray | None = None) -> float:
    """AUC(ViewPosition -> label) in the WEIGHTED training stream.

    This is the dose axis of the dose-response curve, and it is exact rather
    than estimated: both variables are binary, so the AUC follows from the
    2x2 table without fitting anything.

        AUC = P(AP | y=1) P(PA | y=0) + 0.5 [ P(AP|y=1) P(AP|y=0)
                                            + P(PA|y=1) P(PA|y=0) ]

    the usual convention that ties count half. Unweighted on the development
    set this returns 0.706, the documented value of the confounder; at
    `strength = 1` it returns 0.500 by construction. What it answers is how
    much of the association a chosen setting still leaves in the training
    stream, and it answers it in milliseconds instead of 2.3 hours.
    """
    y = np.asarray(y).astype(int)
    vp = np.asarray(vp).astype(str)
    w = np.ones(len(y), float) if w is None else np.asarray(w, float)
    ap = vp == "AP"
    m1, m0 = y == 1, y == 0
    if w[m1].sum() == 0 or w[m0].sum() == 0:
        return float("nan")
    p1 = w[m1 & ap].sum() / w[m1].sum()          # P(AP | positive)
    p0 = w[m0 & ap].sum() / w[m0].sum()          # P(AP | negative)
    return float(p1 * (1 - p0) + 0.5 * (p1 * p0 + (1 - p1) * (1 - p0)))


def balance_report(w: np.ndarray, y: np.ndarray, vp: np.ndarray) -> list[dict]:
    """One row per cell, for printing. Separate so it can be tested."""
    y = np.asarray(y).astype(int)
    vp = np.asarray(vp).astype(str)
    out = []
    for v in sorted(set(vp)):
        for c in sorted(set(y)):
            cell = (vp == v) & (y == c)
            if not cell.any():
                continue
            out.append({"viewpos": v, "target": int(c), "n": int(cell.sum()),
                        "weight": float(w[cell][0])})
    return out


def effective_n(w: np.ndarray) -> float:
    """Kish effective sample size. n itself when all weights are equal.

    This is the honest price tag of the reweighting. It answers the question
    that oversampling always invites: no, drawing an image 2.42 times does not
    turn it into 2.42 patients.
    """
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / np.sum(w ** 2)) if w.size else 0.0


def train_loader_kwargs(weights: np.ndarray | None, seed: int) -> dict:
    """The DataLoader arguments that differ between the two modes.

    Takes the FINISHED weight array rather than the ingredients for it. That
    is not a stylistic choice, it is the fix for a bug that cost 74 minutes of
    compute: this function used to recompute the weights from `y`, `vp` and a
    `strength` argument, `main` printed its table from one call and built the
    sampler from another, and the second call was left without the strength
    parameter. The run announced `strength 0.5` on screen and trained at 1.0.
    The training curve came out bit-identical to the previous run, which is the
    only reason it was caught at all.

    With one array there is one source of truth. The table and the sampler
    cannot disagree, because there is nothing left to disagree about.

    `weights=None` means no balancing: this returns `{"shuffle": True}`, no
    torch generator is created and the global RNG is not advanced. The baseline
    of 0.8166 has to stay reproducible from the same file, so the paired
    comparison differs in the tested quantity and in nothing else.

    `WeightedRandomSampler` is imported here and not at module level because
    `test_rsna_train.py` runs with a stubbed torch when no GPU stack is
    installed, and a top-level import of a name the stub does not carry would
    break the whole test file over a function it never calls.
    """
    if weights is None:
        return {"shuffle": True}
    from torch.utils.data import WeightedRandomSampler
    # An explicit generator, so the draw does not depend on how much torch
    # randomness anything else consumed first.
    g = torch.Generator().manual_seed(seed)
    return {"sampler": WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights), replacement=True, generator=g)}


PERTURBATIONS = {
    "clean":       lambda s: [T.Resize((s, s))],
    "corners":     lambda s: [MaskCorners(), T.Resize((s, s))],
    "zoom_in":     lambda s: [T.Resize((int(s * 1.15),) * 2), T.CenterCrop(s)],
    "shift":       lambda s: [T.Resize((s, s)), T.RandomAffine(0, translate=(0.08, 0.08))],
    "rotate":      lambda s: [T.Resize((s, s)), T.RandomRotation(12)],
    "low_contr":   lambda s: [T.Resize((s, s)), T.ColorJitter(contrast=(0.6, 0.6))],
    "bright":      lambda s: [T.Resize((s, s)), T.ColorJitter(brightness=(1.35, 1.35))],
    "blur":        lambda s: [T.Resize((s, s)), T.GaussianBlur(5, sigma=1.6)],
    "lowres":      lambda s: [T.Resize((int(s * 0.45),) * 2), T.Resize((s, s))],
}


def perturbed_transform(size: int, name: str):
    return T.Compose(PERTURBATIONS[name](size) +
                     [T.Grayscale(num_output_channels=3), T.ToTensor(),
                      T.Normalize(IMNET_MEAN, IMNET_STD)])


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def dml_adapters() -> list[str]:
    """Names of every DirectML adapter, empty list if torch-directml is absent.

    Deliberately separate from pick_device: listing the hardware must be
    possible without also selecting it. rsna_hardware.py prints this, the test
    reads it.
    """
    try:
        import torch_directml
    except ImportError:
        return []
    if not torch_directml.is_available():
        return []
    # .strip(): the driver hands the names back padded with a trailing blank,
    # which then ends up in every CSV cell and turns an equality test on the
    # chip name into a silent mismatch.
    return [str(torch_directml.device_name(i)).strip()
            for i in range(torch_directml.device_count())]


def pick_device(name: str, dml_index: int = 0):
    """Returns (device, pin_memory, label). DirectML is the GPU path here.

    `torch_directml.device()` WITHOUT an index returns adapter 0, and adapter 0
    on this machine is the integrated graphics of the CPU, not the RX 5500 XT
    in the slot. Every run of this project up to 02.08.2026 therefore went to
    the integrated chip, and no log ever said so, because `privateuseone:0`
    names the interface and not the chip. Two changes follow:

      * the index becomes an argument (`--dml-index`), default 0, so that an
        older command line still means exactly what it meant before, and
      * the third return value is a readable label which the caller writes
        into the log AND into results_rsna.csv. Provenance belongs in the
        data, not in a file name; see the checkpoint mix-up of 26.07.

    An index outside the range is a hard stop, not a silent fall back to
    adapter 0. A typo that quietly trains four hours on the wrong chip is
    exactly the failure this function was rewritten to prevent.
    """
    if name in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda"), True, f"cuda:0 {torch.cuda.get_device_name(0)}"
    if name in ("auto", "directml"):
        try:
            import torch_directml
            if torch_directml.is_available():
                names = dml_adapters()
                if not 0 <= dml_index < len(names):
                    listing = "\n".join(f"    {i}  {n}"
                                        for i, n in enumerate(names)) or "    (keiner)"
                    raise SystemExit(
                        f"--dml-index {dml_index} does not exist. "
                        f"Adapters found:\n{listing}")
                return (torch_directml.device(dml_index), False,
                        f"directml:{dml_index} {names[dml_index]}")
        except ImportError:
            if name == "directml":
                raise SystemExit("torch-directml is missing:  pip install torch-directml")
    if name == "directml":
        raise SystemExit("torch-directml finds no device.")
    return torch.device("cpu"), False, "cpu"


class TwoHeadNet(nn.Module):
    """ResNet18 with a second output: a grid x grid field saying WHERE.

    The classification path is the stock `resnet18.forward`, written out step by
    step rather than reimplemented, so it computes exactly what the single
    headed model computes. That is not tidiness, it is the condition for the
    comparison: the two arms have to differ in the head and in nothing else.

    The head taps `layer3`. Two reasons, and the second is the one that
    matters:

      * At 224 pixels layer3 is already 14 by 14, the chosen grid. Nothing in
        the trunk has to be changed to get it. Reaching 14 from layer4 would
        mean changing its stride, and that changes the classification path too,
        which is a second difference between the arms.
      * `adaptive_avg_pool2d` to a FIXED grid means the head keeps its 14 by 14
        whatever the input size. At 512 pixels layer3 delivers 32 by 32 and is
        pooled down. The measuring stick therefore stops moving with the thing
        it measures, which is the whole reason phase 5 comes before phase 8.

    The head is one 1x1 convolution, i.e. a logistic regression per tile on the
    256 layer3 channels. Deliberately the smallest thing that can do the job:
    anything deeper would make "does supervision help" and "does more capacity
    help" the same experiment.
    """

    def __init__(self, grid: int = HEAD_GRID, pretrained: bool = True):
        super().__init__()
        # `pretrained=False` exists for the tests only. They check the wiring,
        # the grid and the identity of the classification path, and none of
        # that needs ImageNet weights. Downloading 45 MB to assert a tensor
        # shape would make the test suite depend on the network.
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 1)
        self.trunk = m
        self.grid = grid
        self.loc = nn.Conv2d(256, 1, kernel_size=1)
        # Zero bias, small weights: the field starts at logit 0, i.e. at
        # probability 0.5 everywhere. With `pos_weight` correcting the tile
        # imbalance that is the neutral start, and lambda is measured from it.
        nn.init.normal_(self.loc.weight, std=0.01)
        nn.init.zeros_(self.loc.bias)

    def features3(self, x):
        m = self.trunk
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        return m.layer3(m.layer2(m.layer1(x)))

    def forward(self, x):
        f3 = self.features3(x)
        f4 = self.trunk.layer4(f3)
        logit = self.trunk.fc(torch.flatten(self.trunk.avgpool(f4), 1))
        field = self.loc(Fn.adaptive_avg_pool2d(f3, self.grid))
        return logit, field


class ClassifierView(nn.Module):
    """The classification branch alone, so Grad-CAM sees a plain classifier.

    `pytorch_grad_cam` expects a module that returns one tensor and it hooks a
    layer by object identity. A two-headed model returns a tuple and breaks it.
    This wrapper hands out the logit and forwards `layer4`, so the SAME
    Grad-CAM code measures both arms. That matters more than it looks: the
    roadmap asks for three numbers, and the informative one is Grad-CAM against
    Grad-CAM, same instrument on both models. Comparing the single-headed
    Grad-CAM with the two-headed head output would compare two instruments and
    call it two models.
    """

    def __init__(self, net: TwoHeadNet):
        super().__init__()
        self.net = net
        self.layer4 = net.trunk.layer4

    def forward(self, x):
        return self.net(x)[0]


def make_model(device, head: bool = False, grid: int = HEAD_GRID):
    if head:
        return TwoHeadNet(grid).to(device)
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m.to(device)


def head_logit(model, x):
    """The classification logit, whichever model this is."""
    out = model(x)
    return (out[0] if isinstance(out, tuple) else out).squeeze(1)


@torch.no_grad()
def predict(model, loader, device, fields: bool = False):
    """Probabilities and labels; with `fields=True` also the head output.

    The field comes back on the CPU as float32, one grid x grid map per image.
    For 3812 validation images at 14 by 14 that is about 3 MB, which is why it
    is simply kept rather than sampled: every later question about the head
    (phase 5b, thresholds, IoU, mAP) can then be answered without retraining,
    the same reason every prediction has gone to disk since phase 3.
    """
    model.eval()
    p, y, f = [], [], []
    for batch in loader:
        x, t = batch[0], batch[1]
        out = model(x.to(device, non_blocking=True))
        logit = (out[0] if isinstance(out, tuple) else out).squeeze(1)
        p.append(torch.sigmoid(logit).float().cpu().numpy())
        y.append(t.numpy())
        if fields:
            if not isinstance(out, tuple):
                raise ValueError("fields=True on a model without a head")
            f.append(torch.sigmoid(out[1][:, 0]).float().cpu().numpy())
    if fields:
        return np.concatenate(p), np.concatenate(y), np.concatenate(f)
    return np.concatenate(p), np.concatenate(y)


def tile_pos_weight(boxes: dict, ids: list[str], grid: int, negatives: str,
                    size: int = 224) -> tuple[float, float]:
    """(pos_weight per tile, mean tile coverage) for the localisation loss.

    Measured on the fitting split of this fold, exactly like the classification
    `pos_weight` two functions down, and for the same reason: an entered number
    is a number nobody checks. It is also NOT the same value for the two
    `--head-negatives` variants, which is easy to miss. With `exclude` only
    annotated images enter and the mean coverage is about 0.118. With `empty`
    all images enter, roughly a fifth of them carry a box, and the mean
    coverage drops by that factor. Using the `exclude` value in the `empty` arm
    would under-correct the imbalance by a factor of four and the head would
    learn to say nothing, which is the exact failure this weight exists to
    prevent.

    Computed from the raw counts even when `--balance-view` is on. That is
    correct rather than sloppy: the reweighting preserves both marginals
    exactly (see `view_balance_weights`), so the prevalence in the stream is
    the prevalence in the split.

    Returns (weight, coverage). A coverage of 0 would be a division by zero and
    means no image in this split has a box, which is a broken split, not a
    weight of infinity.
    """
    use = [i for i in ids if (i in boxes or negatives == "empty")]
    if not use:
        raise ValueError("no image feeds the localisation loss")
    cov = float(np.mean([
        np.asarray(box_mask_pil(boxes.get(i), size), np.float32).mean() / 255.0
        for i in use]))
    if cov <= 0:
        raise ValueError("mean tile coverage is 0, the split carries no boxes")
    return (1.0 - cov) / cov, cov


def loc_loss(field: torch.Tensor, target: torch.Tensor, used: torch.Tensor,
             crit) -> torch.Tensor:
    """Localisation loss, averaged over the images that count.

    `crit` has `reduction="none"`, so the per-tile losses arrive intact and the
    mean is taken here in two steps: over tiles first, then over the images
    with `used == 1`. Letting `BCEWithLogitsLoss` average everything at once
    would silently weight an image by its tile count, which is constant here,
    but it would also average the excluded images in as zeros and quietly scale
    the loss down by whatever share of the batch they are. The scale is not
    cosmetic, lambda is measured from it.

    A batch without a single usable image returns 0 and contributes no
    gradient. With `--head-negatives exclude`, 22.5 percent annotated images
    and batch size 16 that is 0.775^16, about one batch in 59. Over five folds
    the chance that at least one FIRST batch is empty is 8 percent, and the
    first batch is the one lambda is measured from. That is why the measurement
    waits for a batch with a usable image instead of taking batch one.
    """
    per_tile = crit(field[:, 0], target[:, 0])
    per_image = per_tile.flatten(1).mean(1)
    n = used.sum()
    if float(n) == 0:
        return field.sum() * 0.0
    return (per_image * used).sum() / n


def batch_can_set_lambda(used: torch.Tensor) -> bool:
    """May lambda be measured from this batch?

    Only from a batch that has at least one image in the localisation loss.
    Everything else about the measurement stays in the training loop; this is
    the condition alone, split out so it can be tested without a training run.
    """
    return float(used.sum()) > 0


def bce_from_probs(y: np.ndarray, p: np.ndarray, pos_weight: float = 1.0) -> float:
    """The same loss as in training, but computed from probabilities.

    `predict` returns probabilities, not logits. Instead of changing the
    signature (and with it every caller), the loss is recomputed here, with the
    same `pos_weight` as `BCEWithLogitsLoss`. Otherwise the training curve and
    the selection curve are not comparable and the learning curve shows a gap
    between the two that does not exist.

    Clipping at 1e-7 catches p = 0 and p = 1. Without it a single saturated
    prediction returns inf and the whole curve is empty.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean(-(pos_weight * y * np.log(p) + (1 - y) * np.log(1 - p))))


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tpr = np.cumsum(ys) / max(ys.sum(), 1)
    fpr = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
    return float(ps[int(np.argmax(tpr - fpr))])


def scores(y: np.ndarray, p: np.ndarray, thr: float | None = None) -> dict:
    """Ranking metrics plus sensitivity and specificity at a GIVEN threshold.

    `*_oracle` picks the threshold on the same set that is reported on. It is
    written out on purpose as the optimistic counterpart, so that the gap to
    the honest number stays visible.
    """
    out = {"auc": float(roc_auc_score(y, p)),
           "auprc": float(average_precision_score(y, p))}
    t_or = youden_threshold(y, p)
    out["sens_oracle"] = float(((p >= t_or) & (y == 1)).sum() / max((y == 1).sum(), 1))
    out["spec_oracle"] = float(((p < t_or) & (y == 0)).sum() / max((y == 0).sum(), 1))
    t = t_or if thr is None else thr
    out["thr"] = float(t)
    out["sens"] = float(((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1))
    out["spec"] = float(((p < t) & (y == 0)).sum() / max((y == 0).sum(), 1))
    return out


def stratified_scores(y: np.ndarray, p: np.ndarray, vp: np.ndarray,
                      thr: float, thr_by_view: dict[str, float] | None = None
                      ) -> dict:
    """AUC per projection, plus sens/spec at the global and the per-stratum threshold.

    The overall AUC contains the AP/PA effect (header baseline 0.729). Within
    one projection that effect drops out and the baseline is about 0.556. Only
    this stratified number says anything about radiology.

    Why two thresholds on top of that: in the first run the AUC was practically
    identical (0.818 AP against 0.824 PA), yet at ONE threshold sensitivity was
    0.839 in AP and 0.498 in PA. The same number behaves like two different
    tests in the two projections. In PA films half the pneumonias would have
    been missed. That is not a model error but the prevalence difference (0.383
    against 0.093) turning into sensitivity through a fixed threshold. Both are
    reported so the effect stays visible instead of being averaged away.
    """
    out = {}
    for v in ("AP", "PA"):
        m = vp == v
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        pos, neg = y[m] == 1, y[m] == 0
        out[f"auc_{v}"] = float(roc_auc_score(y[m], p[m]))
        out[f"n_{v}"] = int(m.sum())
        out[f"pos_{v}"] = float(y[m].mean())
        out[f"sens_{v}"] = float(((p[m] >= thr) & pos).sum() / max(pos.sum(), 1))
        out[f"spec_{v}"] = float(((p[m] < thr) & neg).sum() / max(neg.sum(), 1))
        if thr_by_view and v in thr_by_view:
            t = thr_by_view[v]
            out[f"thr_{v}"] = float(t)
            out[f"sens_{v}_strat"] = float(((p[m] >= t) & pos).sum() / max(pos.sum(), 1))
            out[f"spec_{v}_strat"] = float(((p[m] < t) & neg).sum() / max(neg.sum(), 1))

    if "auc_AP" in out and "auc_PA" in out:
        # weighted mean of the strata: the AUC that would remain if AP and PA
        # were equally frequent, the confounder-free value
        w = np.array([out["n_AP"], out["n_PA"]], float)
        out["auc_stratified"] = float(
            (out["auc_AP"] * w[0] + out["auc_PA"] * w[1]) / w.sum())
        # The direct measure of the problem: how far apart do the
        # sensitivities of the two projections sit?
        out["sens_gap"] = float(abs(out["sens_AP"] - out["sens_PA"]))
        if "sens_AP_strat" in out and "sens_PA_strat" in out:
            out["sens_gap_strat"] = float(
                abs(out["sens_AP_strat"] - out["sens_PA_strat"]))
    return out


def inner_split(ids: list[str], labels, vp: dict[str, str], seed: int,
                n_splits: int) -> tuple[list[str], list[str]]:
    """Splits fold["train"] into a fit part and a selection part, stratified as outside.

    Picking the checkpoint by AUC on the OUTER val and then reporting that same
    AUC makes every number a maximum over all epochs on the reporting data. On
    Kermany the ceiling hid that. At an AUC of about 0.85 it shifts the number
    by the order of magnitude of the effects one wants to measure.

    Stratification here is again by label x ViewPosition, otherwise the AP/PA
    ratio of the selection split drifts away from val and the threshold no
    longer fits.
    """
    strat = np.array([f"{labels[i]}|{vp[i]}" for i in ids])
    g = np.array(ids)                      # one image per patient
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fit_i, sel_i = next(iter(sgkf.split(np.zeros(len(ids)), strat, g)))
    assert not (set(g[fit_i]) & set(g[sel_i])), "group leak in the inner split!"
    return [ids[i] for i in fit_i], [ids[i] for i in sel_i]


# --------------------------------------------------------------------------
# Grad-CAM against the bounding boxes
# --------------------------------------------------------------------------

def load_boxes(csv_dir: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    df = pd.read_csv(Path(csv_dir) / "stage_2_train_labels.csv")
    df = df[df["Target"] == 1].dropna(subset=["x", "y", "width", "height"])
    out: dict[str, list] = {}
    for pid, x, y, w, h in df[["patientId", "x", "y", "width", "height"]].values:
        out.setdefault(pid, []).append((float(x), float(y), float(w), float(h)))
    return out


def cam_vs_boxes(model, root: Path, ids: list[str], boxes: dict, size: int,
                 n: int, seed: int) -> tuple[dict, pd.DataFrame]:
    """Measures whether the heatmap points at the infiltrate.

    Two measures, both against a chance baseline:

      hit    Does the maximum of the heatmap fall inside a box? ("pointing game")
             Chance baseline = area fraction of the boxes.
      mass   Which share of the heatmap mass lies inside the boxes?
             Chance baseline is again the area fraction.

    Maps that are zero everywhere count as a miss (hit False, mass 0.0) and
    are flagged in the `degenerate` column. See the comment in the loop.

    The area fraction MUST be reported with them. The boxes cover a substantial
    part of the image; a hit rate of 0.6 sounds good and would be next to
    nothing at an area fraction of 0.55. Without that baseline the number is
    worthless, which is the mistake Grad-CAM figures in presentations usually
    make.

    Runs on the CPU: Grad-CAM needs a backward pass through hooks, and that is
    neither fast nor reliable under DirectML.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

    rng = np.random.default_rng(seed)
    pos = [i for i in ids if i in boxes]
    if not pos:
        return {}, pd.DataFrame()
    pick = rng.choice(pos, min(n, len(pos)), replace=False)

    m = model.to("cpu").eval()
    cam = GradCAM(model=m, target_layers=[m.layer4[-1]])
    tf = build_transforms(size, False)
    s = size / BOX_SPACE

    rows = []
    for j, pid in enumerate(pick, 1):
        img = Image.open(root / f"{pid}.png").convert("L")
        x = tf(img).unsqueeze(0)
        heat = cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(1)])[0]
        heat = np.clip(heat, 0, None)

        mask = np.zeros_like(heat, bool)
        for bx, by, bw, bh in boxes[pid]:
            x0, y0 = int(bx * s), int(by * s)
            x1, y1 = int((bx + bw) * s), int((by + bh) * s)
            mask[max(y0, 0):y1, max(x0, 0):x1] = True
        area = float(mask.mean())

        # A map that is zero everywhere after clipping is a FAILURE to
        # localise, not a missing measurement. It used to be skipped, which
        # took the image out of the model's OWN denominator and therefore
        # flattered exactly the model that produces degenerate maps. Measured
        # over five folds at strength 0.5: 3 to 6 of 300 images per fold for
        # the reweighted model, zero for the baseline. It counts as a miss
        # now, and the count is reported so the reader can see how often it
        # happens.
        degenerate = bool(heat.sum() <= 0)
        if degenerate:
            row = {"patientId": pid, "hit": False, "mass": 0.0}
        else:
            yx = np.unravel_index(int(np.argmax(heat)), heat.shape)
            row = {"patientId": pid, "hit": bool(mask[yx]),
                   "mass": float(heat[mask].sum() / heat.sum())}
        row.update({"area": area, "n_boxes": len(boxes[pid]),
                    "degenerate": degenerate})
        rows.append(row)
        if j % 100 == 0:
            print(f"      Grad-CAM {j}/{len(pick)}")

    d = pd.DataFrame(rows)
    if d.empty:
        return {}, d
    res = {
        "cam_n": int(len(d)),
        "cam_degenerate": int(d["degenerate"].sum()),
        "cam_hit": float(d["hit"].mean()),
        "cam_mass": float(d["mass"].mean()),
        "cam_area_baseline": float(d["area"].mean()),
        "cam_hit_lift": float(d["hit"].mean() - d["area"].mean()),
        "cam_mass_lift": float(d["mass"].mean() - d["area"].mean()),
    }
    return res, d


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, default=Path("data/rsna/png512"))
    p.add_argument("--splits", type=Path, default=Path("rsna_splits.json"))
    p.add_argument("--csv", type=Path, default=Path("data/rsna"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=0,
                   help="leave at 0 on Windows: spawn reimports torch in every worker")
    p.add_argument("--inner-splits", type=int, default=6)
    p.add_argument("--balance-view", action="store_true",
                   help="draw training images so that ViewPosition and label "
                        "are independent; see the module header for the "
                        "pre-registered reading of the result")
    p.add_argument("--balance-strength", type=float, default=1.0,
                   help="dose of the correction: 0 is the baseline, 1 full "
                        "independence, values between trade the confounder "
                        "off against the Grad-CAM loss in AP")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "directml", "cpu"])
    p.add_argument("--dml-index", type=int, default=0,
                   help="which DirectML adapter. 0 is the integrated graphics "
                        "and was the silent default of every run before "
                        "02.08.2026, 1 is the RX 5500 XT. The default stays 0 "
                        "so that an old command line keeps its old meaning; "
                        "list the adapters with rsna_hardware.py liste")
    p.add_argument("--head", action="store_true",
                   help="second output: a grid x grid field trained against "
                        "the annotated boxes. Phase 5; see the module header")
    p.add_argument("--head-grid", type=int, default=HEAD_GRID,
                   help="tiles per side. 14 was measured by rsna_kopfraster.py "
                        "and is fixed; the switch exists so the measurement "
                        "can be repeated, not so the value can be shopped for")
    p.add_argument("--head-negatives", default="exclude",
                   choices=["exclude", "empty"],
                   help="images without a box: keep them out of the "
                        "localisation loss, or let them learn an empty field. "
                        "Both are measured, the difference is itself a finding")
    p.add_argument("--head-lambda", type=float, default=0.0,
                   help="weight of the localisation loss. 0 means MEASURE it "
                        "on the first batch that carries an annotated image, "
                        "so both losses start equal, which is the "
                        "pre-registered recipe. A value entered here overrides "
                        "that, skips the range check, and is written into "
                        "results_rsna.csv")
    # Phase 6 turns these two up and nothing else. Defaults are the values of
    # every run up to phase 5, so an unchanged command line means an unchanged
    # experiment.
    p.add_argument("--aug-translate", type=float, default=0.03,
                   help="random shift as a fraction of the edge. Phase 6 uses "
                        "0.08 to make the framing an unreliable cue")
    p.add_argument("--aug-scale", type=float, nargs=2, default=[0.93, 1.07],
                   metavar=("LO", "HI"),
                   help="random rescaling range. Phase 6 uses 0.75 1.0, which "
                        "attacks the same confounder the crop of phase 7 "
                        "attacks, but at the root and without a mask")
    p.add_argument("--aug-degrees", type=float, default=7.0,
                   help="rotation. More is unphysiological on a chest film; "
                        "the switch exists so the value can be seen, not so it "
                        "can be shopped for")
    # Phase 9 turns these two up and nothing else. Until 09.08.2026 they sat
    # hard wired at 0.15 inside TrainTransform while the three above were
    # already arguments.
    p.add_argument("--aug-brightness", type=float, default=0.15,
                   help="photometric jitter, brightness factor drawn per image "
                        "from [1-b, 1+b]. Phase 9 uses 0.60; at the default "
                        "0.15 the jitter removes 4 percent of the global "
                        "brightness cue, see rsna_photometrie_reichweite.py")
    p.add_argument("--aug-contrast", type=float, default=0.15,
                   help="photometric jitter, contrast factor drawn per image "
                        "from [1-c, 1+c]. Phase 9 uses 0.60. This is the knob "
                        "that matters: the global contrast separates AP from "
                        "PA at AUC 0.758, the global brightness at 0.540")
    p.add_argument("--cam-n", type=int, default=300, help="0 = skip Grad-CAM")
    p.add_argument("--out", type=Path, default=Path("results_rsna.csv"))
    p.add_argument("--pred-dir", type=Path, default=Path("predictions_rsna"))
    p.add_argument("--tag", default="",
                   help="suffix for the checkpoint file, e.g. _balview. The "
                        "checkpoint name used to depend on fold and seed "
                        "alone, so every variant silently overwrote the "
                        "previous one; the crop runs took the baseline "
                        "checkpoints with them and nobody noticed.")
    p.add_argument("--history", type=Path, default=None,
                   help="per-epoch history (default: "
                        "<pred-dir>/history_f{fold}_s{seed}.csv)")
    args = p.parse_args()

    # ---- images and boxes have to describe the SAME picture --------------
    # A crop folder written by `rsna_make_crops.py` carries its own
    # `stage_2_train_labels.csv`, with the boxes converted into the grid OF THE
    # CROP. If the images come from such a folder while `--csv` points
    # somewhere else, the model trains on cropped pictures against boxes in the
    # coordinates of the uncropped ones. The head then learns to point at the
    # wrong place, Grad-CAM is scored against rectangles that do not belong to
    # the image, and none of that shows up in the stratified AUC. It would look
    # like a null result.
    #
    # The check is deliberately narrow, so the old path is untouched: with
    # `--images data/rsna/png512 --csv data/rsna` there is no label file inside
    # the image folder and nothing happens.
    eigene_kaesten = args.images / "stage_2_train_labels.csv"
    if eigene_kaesten.exists():
        try:
            gleich = args.csv.resolve() == args.images.resolve()
        except OSError:                       # a path that cannot be resolved
            gleich = str(args.csv) == str(args.images)
        if not gleich:
            print(f"\nABORT: {args.images} carries its own "
                  f"stage_2_train_labels.csv,")
            print(f"       but --csv points at {args.csv}.")
            print("       The boxes would then live in the coordinate frame of")
            print("       a different picture than the pixels. Pass")
            print(f"       --csv {args.images}")
            raise SystemExit(2)

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    sp = json.loads(args.splits.read_text())
    labels = {k: int(v) for k, v in sp["labels"].items()}
    vpmap = sp["viewpos"]
    fold = sp["folds"][args.fold]
    device, pin, dev_label = pick_device(args.device, args.dml_index)

    fit_ids, sel_ids = inner_split(fold["train"], labels, vpmap,
                                   args.seed, args.inner_splits)
    val_ids = fold["val"]
    y_fit = np.array([labels[i] for i in fit_ids])

    print(f"\nFold {args.fold}, Seed {args.seed}, Device {device}")
    # `device` alone prints privateuseone:0 whatever the chip. The label is the
    # only place the chip is named, so it goes into the log AND into the result
    # row further down.
    print(f"  Hardware: {dev_label}")
    # WHICH PIXELS and WHICH BOXES, in the log and not only in the command
    # line. The phase 7 runner greps these two lines: they are the entire lever
    # of that phase, and up to 07.08.2026 they appeared nowhere in the output.
    print(f"  images: {args.images}")
    print(f"  boxes:  {args.csv}")
    print(f"  fit {len(fit_ids)} (pos {y_fit.mean():.3f}) | sel {len(sel_ids)} "
          f"| val {len(val_ids)}")
    print(f"  Targets: overall AUC > 0.729 (header baseline), "
          f"per projection > ~0.556")
    if device.type == "cpu":
        print("  WARNING: CPU. On AMD under Windows:  pip install torch-directml")

    # Reweighting uses the FITTING split of this fold only. Weights taken from
    # the whole development set would carry information out of the other folds.
    vp_fit = np.array([vpmap.get(i, "?") for i in fit_ids])
    # Computed ONCE. Everything below, the printed table and the sampler alike,
    # reads this one array; see train_loader_kwargs for why that matters.
    w_fit = (view_balance_weights(y_fit, vp_fit, args.balance_strength)
             if args.balance_view else None)
    if args.balance_view:
        print(f"\n  --balance-view at strength {args.balance_strength:g}: "
              f"ViewPosition and label decoupled in the training stream")
        print(f"    {'projection':<12}{'label':>7}{'n':>8}{'draws/image':>14}")
        for r in balance_report(w_fit, y_fit, vp_fit):
            flag = "   <-- small stratum" if r["n"] < 50 else ""
            print(f"    {r['viewpos']:<12}{r['target']:>7}{r['n']:>8}"
                  f"{r['weight']:>13.2f}x{flag}")
        print(f"    effective sample size {effective_n(w_fit):.0f} of "
              f"{len(w_fit)} (Kish)")
        # The dose, exact and computed before a single epoch is spent.
        print(f"    AUC(ViewPosition -> label) in the stream: "
              f"{residual_view_label_auc(y_fit, vp_fit, w_fit):.3f}   "
              f"(untouched {residual_view_label_auc(y_fit, vp_fit):.3f}, "
              f"fully decoupled 0.500)")
        print("    PRIMARY endpoint: AUC(score -> ViewPosition) must fall "
              "from 0.8166 +- 0.0098.")
        print("    The RAW AUC is expected to fall as well. That is "
              "the success signature, not a regression.")
        print("    Predicted before the runs: about 0.044. Measured "
              "over five folds: 0.0151 +- 0.0061 at strength 0.5, "
              "0.0233 +- 0.0021 at strength 1.0.")
        print("    See the module header.")

    # Boxes are loaded BEFORE training now, not after it. Up to phase 4 they
    # were read once the model was finished, purely to score Grad-CAM; from
    # phase 5 they are training data. That single line is the whole change of
    # target, and it is also the moment the hit rate stops being an
    # independent control. See the module header of `zielgroesse_lokalisation`.
    boxes = load_boxes(args.csv) if args.head else None
    tile_w, tile_cov = ((None, None) if not args.head else
                        tile_pos_weight(boxes, fit_ids, args.head_grid,
                                        args.head_negatives, args.size))
    if args.head:
        n_boxed = sum(1 for i in fit_ids if i in boxes)
        print(f"\n  --head: {args.head_grid} x {args.head_grid} field, "
              f"negatives '{args.head_negatives}'")
        print(f"    {n_boxed} of {len(fit_ids)} fitting images carry a box")
        print(f"    mean tile coverage {tile_cov:.4f} -> pos_weight per tile "
              f"{tile_w:.2f}")
        print("    PRE-REGISTERED: the primary endpoint of phase 5 is A, the "
              "stratified AUC.")
        print("    B is not an endpoint here. A model trained to point will "
              "point better;")
        print("    that is a definition, not a finding. The question is what "
              "it COSTS.")
        print("    Smoke test before any comparison: the head has to beat the "
              "LOCATION PRIOR,")
        print("    not chance. rsna_kopf_auswertung.py checks that.")

    # TrainTransform, not build_transforms(size, True): the augmentation has to
    # move image and box mask together. It runs in BOTH arms so the two differ
    # in the head alone; see the box trap in the module header.
    train_tf = TrainTransform(args.size, args.aug_translate,
                              tuple(args.aug_scale), args.aug_degrees,
                              args.aug_brightness, args.aug_contrast)
    # Into the log, because a strength that lives only in a command line is a
    # strength that gets lost. The phase 6 runner greps this line.
    print(f"\n  --aug: rotation {args.aug_degrees:g} deg, translate "
          f"{args.aug_translate:.3f}, scale {args.aug_scale[0]:.2f} to "
          f"{args.aug_scale[1]:.2f}")
    print(f"  --aug photometrisch: brightness {args.aug_brightness:.2f}, "
          f"contrast {args.aug_contrast:.2f}")
    if (args.aug_translate, tuple(args.aug_scale)) != (0.03, (0.93, 1.07)):
        print("    STRONGER THAN THE DEFAULT. The box mask moves with the "
              "image, and")
        print("    a mistake there now weighs more. tests/test_rsna_kopf.py "
              "checks it")
        print("    at exactly these numbers.")
    tr = DataLoader(RsnaDataset(args.images, fit_ids, labels, train_tf,
                                boxes=boxes, grid=args.head_grid,
                                negatives=args.head_negatives, size=args.size),
                    batch_size=args.batch, num_workers=args.workers,
                    pin_memory=pin, drop_last=True,
                    **train_loader_kwargs(w_fit, args.seed))
    # Not the switch, the draw. Measured on the transform object the loader
    # above actually holds, so it also catches the case where the switch is
    # read and then not wired through; see the docstring and phase 8's
    # `input_px`. The abort is hard for the same reason it is hard there: a
    # mismatch means the row this run writes would claim a strength the run did
    # not use, and every paired comparison built on it would be wrong in a way
    # nobody could find afterwards.
    b_gemessen, c_gemessen = gemessene_jitter_staerke(tr.dataset.tf)
    print(f"    gemessen am Ziehen: brightness {b_gemessen:.3f}, contrast "
          f"{c_gemessen:.3f}  (512 Ziehungen, erwartet je der Schalterwert)")
    for name, soll, ist in (("brightness", args.aug_brightness, b_gemessen),
                            ("contrast", args.aug_contrast, c_gemessen)):
        if abs(ist - soll) > max(0.25 * soll, 0.03):
            raise SystemExit(
                f"ABORT: --aug-{name} says {soll:.3f}, the transform the "
                f"loader holds draws {ist:.3f}. The switch is not reaching "
                f"the jitter.")
    sel = DataLoader(RsnaDataset(args.images, sel_ids, labels,
                                build_transforms(args.size, False)),
                     batch_size=args.batch * 2, num_workers=args.workers)
    va = DataLoader(RsnaDataset(args.images, val_ids, labels,
                               build_transforms(args.size, False)),
                    batch_size=args.batch * 2, num_workers=args.workers)

    model = make_model(device, head=args.head, grid=args.head_grid)
    # Positive rate 0.225. The imbalance tips the other way than on Kermany
    # (0.74), so pos_weight is > 1 instead of < 1.
    pos_weight = torch.tensor([(y_fit == 0).sum() / max((y_fit == 1).sum(), 1)],
                              dtype=torch.float32, device=device)
    print(f"  pos_weight {pos_weight.item():.2f}")
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    crit_loc = (nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([tile_w], dtype=torch.float32, device=device),
        reduction="none") if args.head else None)
    # 0 means "measure it". It is filled in below, printed, and written to
    # results_rsna.csv together with the batch it came from, so both can be
    # read off a finished run instead of being reconstructed from the command
    # line. 0 in `lam_batch` means it was not measured but entered by hand.
    lam = float(args.head_lambda)
    lam_batch = 0
    # 0 means "not measured yet". Filled from the first training batch below.
    # WHY A MEASURED EDGE LENGTH AND NOT JUST args.size, added 08.08.2026 for
    # phase 8: `--size` is the ONE lever of that phase, and up to this line it
    # appeared nowhere in the output. An arm that had silently trained at 224
    # would produce a clean looking null result, exactly the phase 7 problem
    # with `--images` one level further in.
    #
    # `args.size` is a self report: it says what the run believed it was doing.
    # This one is not: it is the edge length of the tensor the model actually
    # receives, so it also catches the case where the switch is read and then
    # not wired through. That has happened in this project before, see
    # `--balance-strength`, which was computed correctly and then passed on as
    # the default.
    input_px = 0
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(len(tr), 1))

    # Per-epoch history. A single result row at the end leaves no learning
    # curve to show, and a run that dies in hour four leaves nothing at all.
    #
    # What is logged is the SELECTION split (sel), not the reporting set (val).
    # That is the point rather than a saving in the wrong place: sel is exempt
    # from fitting, so the gap between training loss and sel loss shows the
    # overfitting in full. A curve on val would also be an invitation to pick
    # the epoch afterwards, the circular reasoning this project avoids
    # everywhere else.
    hist_path = (args.history if args.history is not None else
                 args.pred_dir / f"history_f{args.fold}_s{args.seed}.csv")
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []

    def write_history() -> None:
        """Written after EVERY epoch, not at the end. An aborted 24-hour run
        should still leave its curve behind."""
        pd.DataFrame(history).to_csv(hist_path, index=False)

    # The checkpoint used to be written once, after training, Grad-CAM and all
    # the perturbations. A machine that goes to sleep in hour two therefore
    # threw away every trained weight and left only the curve. It is written
    # after every improvement now: 45 MB against 74 minutes is not a trade
    # worth thinking about.
    ckpt = Path(f"checkpoints/rsna_f{args.fold}_s{args.seed}{args.tag}.pth")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if ckpt.exists():
        print(f"  NOTE: {ckpt} exists "
              f"({(time.time() - ckpt.stat().st_mtime) / 3600:.1f} h old) and "
              f"is being replaced. Use --tag to keep variants apart.")

    best_sel, best_state, best_ep = -1.0, None, -1
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        # Read the LR BEFORE the epoch. OneCycleLR advances after every step,
        # so after the loop it would hold the rate of the next epoch and the
        # curve would be shifted by one epoch.
        lr_now = float(sched.get_last_lr()[0])
        # Accumulate on the device and fetch it ONCE per epoch. `float(loss)`
        # inside the loop would wait on the device at every batch, roughly 480
        # synchronisation points per epoch just to record one number for the
        # curve. `drop_last=True` makes all batches the same size, so the mean
        # of the means is the right one.
        loss_sum = torch.zeros((), device=device)
        loc_sum = torch.zeros((), device=device)
        n_batch = 0
        for batch in tr:
            x = batch[0].to(device, non_blocking=True)
            t = batch[1].to(device, non_blocking=True)
            # Measured once, on the first batch of the run, and then compared
            # against the switch. The abort is deliberate and it is hard: a
            # mismatch here means the resolution of this run is not the
            # resolution its result row will claim, and every paired comparison
            # built on that row would be wrong in a way nobody could find
            # afterwards. Better a run that dies in the first minute.
            if input_px == 0:
                input_px = int(x.shape[-1])
                print(f"  input {input_px} x {int(x.shape[-2])} px, measured on "
                      f"the first batch (--size {args.size})")
                if input_px != int(args.size) or int(x.shape[-2]) != int(args.size):
                    raise SystemExit(
                        f"ABORT: --size says {args.size}, the model receives "
                        f"{int(x.shape[-2])} x {input_px}. The switch is not "
                        f"reaching the transform.")
            opt.zero_grad(set_to_none=True)
            if not args.head:
                loss = crit(model(x).squeeze(1), t)
                l_loc = torch.zeros((), device=device)
            else:
                target = batch[2].to(device, non_blocking=True)
                used = batch[3].to(device, non_blocking=True).float()
                logit, field = model(x)
                l_cls = crit(logit.squeeze(1), t)
                l_loc = loc_loss(field, target, used, crit_loc)
                # THE ONE PLACE lambda is decided, and it is decided by
                # measurement: scale the localisation loss so that both start
                # at the same size. "Equally weighted" is then a fact about
                # this run rather than a claim about the intention, and it is
                # reproducible instead of guessed. Fixed after this batch and
                # never touched again, because a lambda that keeps adjusting
                # would make the two losses trade places during training and no
                # comparison would mean anything.
                #
                # The condition is `used.sum() > 0`, not "this is batch one".
                # With --head-negatives exclude an image without a box
                # contributes nothing, so a batch that holds no annotated image
                # returns a localisation loss of exactly zero and the ratio
                # becomes whatever the clamp at 1e-8 makes of it. See loc_loss
                # for how often that is.
                #
                # Waiting for the next batch changes nothing else. The skipped
                # batch has a localisation loss of exactly zero, so
                # `l_cls + lam * l_loc` equals `l_cls` whatever lam holds, and
                # its gradient is the same either way. For a first batch that
                # does carry an annotated image this branch behaves exactly as
                # it did before, which is what keeps folds comparable.
                if lam == 0.0 and batch_can_set_lambda(used):
                    lam = float((l_cls.detach() / l_loc.detach().clamp(min=1e-8)).cpu())
                    lam_batch = n_batch + 1
                    print(f"    lambda measured on batch {lam_batch}: "
                          f"{lam:.4f}  "
                          f"(classification {float(l_cls.detach()):.4f}, "
                          f"localisation {float(l_loc.detach()):.4f})")
                    if lam_batch > 1:
                        print(f"    batch 1 carried no annotated image, so the "
                              f"measurement moved to batch {lam_batch}. The "
                              f"batches before it gave the head no gradient.")
                    if not LAMBDA_MIN <= lam <= LAMBDA_MAX:
                        raise SystemExit(
                            f"ABORT: lambda {lam:.4g} lies outside "
                            f"[{LAMBDA_MIN:g}, {LAMBDA_MAX:g}]. The two losses "
                            f"do not start at the same size, so this run would "
                            f"not be comparable with the others. Nothing has "
                            f"been written yet.")
                loss = l_cls + lam * l_loc
            loss.backward()
            opt.step(); sched.step()
            loss_sum += loss.detach(); loc_sum += l_loc.detach(); n_batch += 1
        # An epoch in which not one batch carried an annotated image would
        # leave lambda at 0, the head would get no gradient at all, and the run
        # would look normal to the end. It cannot happen with these splits, but
        # a silent zero is the failure mode this whole block exists against.
        if args.head and lam == 0.0:
            raise SystemExit(
                "ABORT: no batch of this epoch carried an annotated image, so "
                "lambda was never measured and the head never trained.")
        train_loss = float(loss_sum.cpu()) / max(n_batch, 1)
        train_loc = float(loc_sum.cpu()) / max(n_batch, 1)
        ps, ys = predict(model, sel, device)
        a = roc_auc_score(ys, ps)
        improved = a > best_sel
        if improved:
            best_sel, best_ep = a, ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            # Write through a temporary file and rename. A power cut in the
            # middle of torch.save would otherwise leave a truncated
            # checkpoint, which is worse than none: it loads far enough to
            # look plausible.
            tmp = ckpt.with_suffix(".pth.tmp")
            torch.save(best_state, tmp)
            tmp.replace(ckpt)
        dt = time.time() - t0
        history.append({
            "fold": args.fold, "seed": args.seed, "epoch": ep + 1,
            "train_loss": train_loss,
            # Logged separately, because the combined loss cannot show whether
            # the head is learning at all. A localisation loss that sits still
            # while the total falls is the box trap showing itself in the only
            # place it ever shows.
            "train_loc_loss": train_loc if args.head else float("nan"),
            "sel_loss": bce_from_probs(ys, ps, float(pos_weight.item())),
            "sel_auc": float(a),
            "lr": lr_now, "sec": dt,
            "is_best": int(improved),
        })
        write_history()
        print(f"  epoch {ep + 1}/{args.epochs}  sel AUC {a:.4f}  "
              f"train loss {history[-1]['train_loss']:.4f}  "
              f"sel loss {history[-1]['sel_loss']:.4f}"
              f"{f'  loc {train_loc:.4f}' if args.head else ''}"
              f"{'  <-- best so far' if improved else ''}  "
              f"[{dt:.0f}s, ~{dt * (args.epochs - ep - 1) / 60:.0f} min left]")

    p_last, y = predict(model, va, device)
    auc_last = float(roc_auc_score(y, p_last))

    model.load_state_dict(best_state)
    p_sel, y_sel = predict(model, sel, device)
    thr = youden_threshold(y_sel, p_sel)          # threshold NOT from the reporting set
    # ... and per projection likewise on the selection split, not on val. A
    # per-stratum threshold searched on the reporting set would be the same
    # circular reasoning as the global one and would overstate the gain.
    vp_sel = np.array([vpmap[i] for i in sel_ids])
    thr_by_view = {v: youden_threshold(y_sel[vp_sel == v], p_sel[vp_sel == v])
                   for v in ("AP", "PA")
                   if (vp_sel == v).sum() >= 50
                   and len(np.unique(y_sel[vp_sel == v])) > 1}
    if args.head:
        p_val, y, val_fields = predict(model, va, device, fields=True)
    else:
        p_val, y = predict(model, va, device)
        val_fields = None

    vp = np.array([vpmap[i] for i in val_ids])
    res = scores(y, p_val, thr)
    res.update(stratified_scores(y, p_val, vp, thr, thr_by_view))
    res.update({"fold": args.fold, "seed": args.seed, "epochs": args.epochs,
                "auc_last": auc_last, "auc_sel": float(best_sel),
                "best_epoch": best_ep + 1, "n_fit": len(fit_ids),
                "n_sel": len(sel_ids), "n_val": len(val_ids),
                # Which chip computed this row. Empty in every row written
                # before 02.08.2026, and empty there means adapter 0, the
                # integrated graphics. Two rows may only be compared when
                # these two fields agree; see erklaerungen/12_hardwarewechsel.md.
                "device_name": dev_label,
                "dml_index": (args.dml_index if str(device).startswith("privateuseone")
                              else -1),
                # WHICH ARM this row is, and which files carry it. Until phase 5
                # the tag lived only in the checkpoint FILE NAME, so from a row
                # in results_rsna.csv there was no way back to its weights. With
                # three arms writing into one CSV that stops being a detail: the
                # arms are otherwise only told apart indirectly, through the
                # combination of head, head_negatives, balance_view and
                # dml_index, and that combination stops being unique the moment
                # a fourth arm repeats it.
                #
                # Same lesson as `dml_index` right above: provenance that lives
                # only in a command line is provenance that gets lost. This
                # project has already overwritten five checkpoints unnoticed.
                #
                # Empty in every row written before 04.08.2026. Empty there
                # means the default tag, so predictions_rsna/ for the old rows.
                "tag": args.tag,
                "pred_dir": str(args.pred_dir),
                "ckpt": str(ckpt),
                # WHICH PIXELS and WHICH BOXES this row was trained on. Added
                # 07.08.2026 for phase 7, for the same reason that put
                # `dml_index`, `tag` and the four `aug_*` columns here:
                # provenance that lives only in a command line is provenance
                # that gets lost.
                #
                # Phase 7 has exactly ONE lever and it is these two paths. An
                # arm that accidentally trained on png512 would produce a clean
                # looking null result, and nothing anywhere in the output would
                # contradict it. That is the phase 6 `staerke_pruefen` problem
                # one level deeper: there the lever at least had four columns.
                #
                # Empty in every row written before 07.08.2026, and empty there
                # means the defaults, data/rsna/png512 and data/rsna.
                "images": str(args.images),
                "csv": str(args.csv),
                # WHICH RESOLUTION, added 08.08.2026 for phase 8, and for the
                # same reason as the two lines above: `results_rsna.csv` had 76
                # columns and not one of them named the edge length. `size` is
                # the switch, `input_px` is what the model actually got, and
                # the run aborts if they disagree. Empty in every row written
                # before 08.08.2026, and empty there means 224.
                "size": int(args.size),
                "input_px": int(input_px),
                # Goes into results_rsna.csv so that a row can never be
                # mistaken for a baseline row later on.
                "balance_view": int(args.balance_view),
                "balance_strength": (float(args.balance_strength)
                                     if args.balance_view else 0.0),
                # Both derived from the SAME w_fit the sampler used, not
                # recomputed. A recomputation is how the strength argument got
                # lost in the first place.
                "balance_residual_auc": residual_view_label_auc(y_fit, vp_fit,
                                                                w_fit),
                "n_fit_effective": (effective_n(w_fit) if w_fit is not None
                                    else float(len(fit_ids))),
                # The head, in the data rather than in a file name. Same
                # lesson as `dml_index`: provenance that lives only in a
                # command line is provenance that gets lost.
                "head": int(args.head),
                "head_grid": int(args.head_grid) if args.head else 0,
                "head_negatives": args.head_negatives if args.head else "",
                # The augmentation strength, in the data and not only in the
                # command line. Old rows leave these empty, and empty means the
                # default, which is what every run up to phase 5 used.
                "aug_translate": float(args.aug_translate),
                "aug_scale_lo": float(args.aug_scale[0]),
                "aug_scale_hi": float(args.aug_scale[1]),
                "aug_degrees": float(args.aug_degrees),
                # THE PHOTOMETRIC STRENGTH, added 09.08.2026 for phase 9. Two
                # switches and two MEASURED values, the same pairing as
                # `size`/`input_px`: the first says what the run believed it
                # was doing, the second is the spread of the factors the
                # transform in the loader really drew. Empty in every row
                # written before 09.08.2026, and empty there means 0.15.
                "aug_brightness": float(args.aug_brightness),
                "aug_contrast": float(args.aug_contrast),
                "aug_brightness_measured": float(b_gemessen),
                "aug_contrast_measured": float(c_gemessen),
                "head_lambda": float(lam) if args.head else float("nan"),
                "head_lambda_measured": int(args.head and args.head_lambda == 0.0),
                # Which batch it came from. 1 is the normal case, 0 means the
                # value was entered by hand, and anything above 1 says the
                # first batch held no annotated image. Without this column that
                # fact lives only in a log line.
                "head_lambda_batch": int(lam_batch) if args.head else 0,
                "head_tile_pos_weight": float(tile_w) if args.head else float("nan"),
                "head_tile_coverage": float(tile_cov) if args.head else float("nan")})

    # The primary endpoint, computed here as well so it is visible in the run
    # instead of only after rsna_crop_compare.py. Same definition as
    # `rsna_crop_compare.score_to_view`, deliberately not folded to
    # max(a, 1 - a): the direction carries meaning and 0.5 is the floor of the
    # channel, not of the number.
    m_vp = np.isin(vp, ["AP", "PA"])
    res["auc_view"] = (float(roc_auc_score((vp[m_vp] == "AP").astype(int),
                                           p_val[m_vp]))
                       if m_vp.sum() and len(set(vp[m_vp])) > 1 else float("nan"))

    print(f"\n  AUC overall     {res['auc']:.4f}   (last epoch {auc_last:.4f}, "
          f"header baseline 0.729)")
    for v in ("AP", "PA"):
        if f"auc_{v}" in res:
            print(f"  AUC {v} only     {res[f'auc_{v}']:.4f}   "
                  f"(n={res[f'n_{v}']}, pos={res[f'pos_{v}']:.3f}, "
                  f"baseline ~0.556)")
    if "auc_stratified" in res:
        print(f"  AUC stratified  {res['auc_stratified']:.4f}  <-- the honest number")
    if res["auc_view"] == res["auc_view"]:            # not NaN
        d = res["auc_view"] - 0.8166
        print(f"  AUC score->view {res['auc_view']:.4f}   ({d:+.4f} against the "
              f"baseline 0.8166 +- 0.0098)  <-- PRIMARY endpoint, must fall")
    print(f"  Sens {res['sens']:.3f} / Spec {res['spec']:.3f} "
          f"(oracle {res['sens_oracle']:.3f}/{res['spec_oracle']:.3f})")

    # The core question: does ONE threshold behave the same in both projections?
    if "sens_gap" in res:
        print(f"\n  Threshold           {'global':>22}   {'per projection':>22}")
        for v in ("AP", "PA"):
            g = f"Sens {res[f'sens_{v}']:.3f} Spez {res[f'spec_{v}']:.3f}"
            s = (f"Sens {res[f'sens_{v}_strat']:.3f} Spez {res[f'spec_{v}_strat']:.3f}"
                 f" @{res[f'thr_{v}']:.3f}" if f"sens_{v}_strat" in res else "-")
            print(f"    {v:<16}{g:>22}   {s:>22}")
        line = f"    {'Sens-Luecke':<16}{res['sens_gap']:>22.3f}"
        if "sens_gap_strat" in res:
            line += f"   {res['sens_gap_strat']:>22.3f}"
        print(line)
        print("    A fixed threshold at unequal prevalence (0.383 vs 0.093) is")
        print("    effectively a different test in the two projections.")

    # ---- Perturbations, above all the corner ablation ------------------
    preds = {"patientId": list(val_ids), "y": y.tolist(), "viewpos": vp.tolist(),
             "p_clean": p_val.tolist(), "p_last_epoch": p_last.tolist()}
    print()
    for name in [n for n in PERTURBATIONS if n != "clean"]:
        torch.manual_seed(args.seed); random.seed(args.seed)
        ds = RsnaDataset(args.images, val_ids, labels,
                         perturbed_transform(args.size, name))
        pp, yy = predict(model, DataLoader(ds, batch_size=args.batch * 2,
                                           num_workers=args.workers), device)
        res[f"auc_{name}"] = float(roc_auc_score(yy, pp))
        preds[f"p_{name}"] = pp.tolist()
        tag = "  <-- marker ablation" if name == "corners" else ""
        print(f"  Perturbation {name:<10} AUC {res[f'auc_{name}']:.4f}  "
              f"({res[f'auc_{name}'] - res['auc']:+.4f}){tag}")

    # ---- Grad-CAM gegen Bounding Boxes ---------------------------------
    cam_df = pd.DataFrame()
    if args.cam_n:
        print(f"\n  Grad-CAM on {args.cam_n} positive val images (CPU)...")
        cam_boxes = boxes if boxes is not None else load_boxes(args.csv)
        # The two-headed model goes in through ClassifierView, so Grad-CAM sees
        # the same plain classifier it sees in the other arm. This is the
        # middle row of the three-way table in the roadmap and the only one
        # that compares two MODELS rather than two INSTRUMENTS.
        cam_model = ClassifierView(model) if args.head else model
        cam_res, cam_df = cam_vs_boxes(cam_model, args.images, val_ids, cam_boxes,
                                       args.size, args.cam_n, args.seed)
        res.update(cam_res)
        if cam_res:
            print(f"  Hit rate {cam_res['cam_hit']:.3f}  vs chance "
                  f"{cam_res['cam_area_baseline']:.3f}  "
                  f"(margin {cam_res['cam_hit_lift']:+.3f})")
            print(f"  Mass     {cam_res['cam_mass']:.3f}  vs chance "
                  f"{cam_res['cam_area_baseline']:.3f}  "
                  f"(margin {cam_res['cam_mass_lift']:+.3f})")
            print(f"  Degenerate maps {cam_res['cam_degenerate']} of "
                  f"{cam_res['cam_n']}, counted as a miss (see cam_vs_boxes).")
            print("  Without the chance baseline the hit rate means nothing:")
            print("  the boxes cover a substantial part of the image.")

    # ---- Saving ---------------------------------------------------------
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds).to_csv(
        args.pred_dir / f"rsna_f{args.fold}_s{args.seed}.csv", index=False)
    # The selection predictions are written as well. Without them any later
    # question about the threshold can only be answered as an oracle (threshold
    # searched on the reporting set = too optimistic).
    pd.DataFrame({"patientId": sel_ids, "y": y_sel.tolist(),
                  "viewpos": vp_sel.tolist(), "p_sel": p_sel.tolist()}).to_csv(
        args.pred_dir / f"sel_f{args.fold}_s{args.seed}.csv", index=False)
    if not cam_df.empty:
        cam_df.to_csv(args.pred_dir / f"cam_f{args.fold}_s{args.seed}.csv", index=False)
    if val_fields is not None:
        # Every validation image, not a sample: 3812 maps of 14 by 14 in float32
        # are about 3 MB. Sampling here would mean retraining for every later
        # question, and phase 5b (thresholds, connected tiles, IoU, mAP) is
        # nothing but later questions about exactly this array.
        np.savez_compressed(
            args.pred_dir / f"head_f{args.fold}_s{args.seed}.npz",
            patientId=np.array(val_ids), field=val_fields.astype(np.float32),
            grid=np.int32(args.head_grid))
    # Already on disk from the epoch loop; rewritten here only so the file is
    # certainly the selected state even if the loop never improved.
    torch.save(best_state, ckpt)

    # Read, merge and rewrite instead of appending. A plain mode="a" writes
    # the values in the order of the CURRENT run under a header that came from
    # an earlier one. As soon as the metric set changes, and thr_AP,
    # sens_AP_strat and the rest belong to it, 49 values sit under 41 column
    # names. The file is then not broken in the sense of unreadable, but worse:
    # silently shifted.
    row = pd.DataFrame([res])
    if args.out.exists():
        old = pd.read_csv(args.out)
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(args.out, index=False)
    print(f"\nsaved: {args.out}, {args.pred_dir}/, {ckpt}")


if __name__ == "__main__":
    main()
