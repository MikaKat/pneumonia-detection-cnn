"""Preprocessing stage images for the web app.

The web app shows the pipeline as a chain: the uploaded image, then one tile per
processing step, each connected by an arrow, ending in the Grad-CAM heatmap.
This module produces the picture for each tile. `main.py` calls the steps one by
one and publishes each result as soon as it exists, so the chain fills up in the
browser while the analysis is still running.

Honesty about what is shown
---------------------------
The deployed classifier runs on the FULL image, exactly as it was trained. Two
kinds of tile sit off to the side of that path and neither one touches the
score.

The lung mask was built and measured in this project (`segmentation/`,
`rsna/pipeline/rsna_make_crops.py`) and rejected on the RSNA data: the
pixel-exact mask destroys pathology and re-encodes the shape as a shortcut, and
the crop did not move the confounder score in the intended direction. It is
computed here for real and shown for real, but flagged as `explored` so nobody
reads the chain as "this is what the score came out of".

The head field is different in kind. It is the second output of the very
network that produced the score, so it is not rejected work; it simply does not
feed the number. It is shown without a box and without a cut-off because its
level is uncalibrated. Both are `group: "aside"`; the frontend draws them on a
separate branch rather than as links in the chain.

Every image leaves here as a base64 PNG without the `data:` prefix, capped at
`VIEW_SIZE` on the long edge so the JSON stays small.
"""

from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

VIEW_SIZE = 384          # long edge of every stage image sent to the browser
SEG_SIZE = 256           # the resolution the U-Net was trained at
CROP_PAD = 0.05          # margin around the lung bounding box, per side


# --------------------------------------------------------------------------
# The chain, as static metadata
# --------------------------------------------------------------------------
# The frontend fetches this once (GET /api/pipeline) and can draw the empty
# chain with all its labels before a single image has been uploaded. Keys must
# match the `key` of the stages emitted during the run.
# The wording of the model input stage used to branch on which training run the
# weights came from. Since phase 10 there is only one deployed model, so the
# branch is gone and the text says plainly what the transform does. It has to
# keep saying it: main.py rebuilds that transform from
# rsna_train.build_transforms, and a caption describing a different one would be
# a plain untruth.
_MODEL_INPUT_TEXT = {
    "caption": "Greyscale, resize to 224x224, then the fixed ImageNet "
               "normalisation the model was trained with. Computed on the full "
               "uploaded image.",
    "note": "The normalisation subtracts the same constants from every image, so "
            "unlike a per-image standardisation it lets brightness differences "
            "between images survive. It is invisible here anyway: the tensor is "
            "re-stretched to 0-255 for display, and normalisation and stretch are "
            "both affine, so they cancel. What you can see is the resize. The model "
            "works on the unbounded values.",
}


# This list is the scored path and nothing else. Every entry here really does
# feed the next one, so the arrows the frontend draws between them are true.
# Work that was built, measured and left out lives in ASIDES below and is drawn
# somewhere else entirely: a badge inside a row of arrows loses against the
# arrows, and readers reasonably concluded that the crop fed the classifier.
PIPELINE = [
    {
        "key": "upload",
        "title": "Uploaded image",
        "caption": "The chest X-ray as received, converted to RGB.",
        "status": "active",
    },
    {
        "key": "model_input",
        # Named after what it does. "What the model sees" said nothing about the
        # two operations that actually change the picture, and standing next to a
        # crop tile it invited the reading that the crop was what got seen.
        #
        "title": "Resize and normalise",
        "caption": _MODEL_INPUT_TEXT["caption"],
        "status": "active",
        "note": _MODEL_INPUT_TEXT["note"],
    },
    {
        "key": "heatmap",
        "title": "Grad-CAM heatmap",
        "caption": "Where the last convolutional block contributed most to the score, "
                   "averaged over the five models of the ensemble.",
        "status": "active",
        "note": "One map per fold would explain a model that did not produce the "
                "number shown. Each map is already stretched to 0-1 by Grad-CAM, so "
                "the average is the share of the five models that find a place warm. "
                "As a pointer this map is weak: measured against the radiologist "
                "boxes it reaches 0.73, below the 0.75 of a fixed template that "
                "ignores the image and marks where pneumonia usually sits. It is "
                "kept because it is the only map that comes from the score itself.",
    },
]

# --------------------------------------------------------------------------
# Off to the side
# --------------------------------------------------------------------------
# Computed on every image and shown, but not part of the scored path and not
# drawn as a link in it. The segmenter is a separate piece of work that does its
# own job well; the honest way to show it is on its own, not as a step of a
# chain it does not belong to.
ASIDES = [
    {
        "key": "mask",
        "title": "Lung finder",
        "caption": "A U-Net trained from scratch on Montgomery/Shenzhen segments both "
                   "lungs at 256x256; the mask is then cleaned morphologically.",
        "status": "explored",
        "group": "aside",
        "note": "This runs on your image, but it does not touch the score. Feeding the "
                "classifier only the lungs was measured and rejected: the silhouette "
                "alone still carries the confounder, and the mask cuts into the "
                "pathology it is meant to isolate. It is shown because it is a working "
                "segmenter, not because the classifier needs it.",
    },
    {
        "key": "head_field",
        "title": "Where the model points",
        "caption": "The second output of the same network: a 14x14 field trained "
                   "against the radiologist boxes, averaged over the five models.",
        "status": "active",
        "group": "aside",
        "note": "This comes out of the very network that produced the score, but it "
                "does not feed the score. It is the stronger of the two maps by a "
                "wide margin: 0.91 against the radiologist boxes, where Grad-CAM "
                "reaches 0.73 and a fixed anatomical template reaches 0.75. It is "
                "drawn as a gradient with no box and "
                "no cut-off, on purpose: the level of this field is not calibrated. "
                "On images without pneumonia it still lights up in 62 % of cases, so "
                "a drawn box would be a claim about exactly the quantity that was "
                "measured and found wanting. Read it as a hint about the region, "
                "never as a finding.",
    },
]

PIPELINE_BY_KEY = {s["key"]: s for s in [*PIPELINE, *ASIDES]}


def stage(key: str, image_b64: str | None, **extra) -> dict:
    """Assemble one stage message from its static metadata plus the image."""
    meta = PIPELINE_BY_KEY[key]
    out = {
        "key": key,
        "title": meta["title"],
        "caption": meta["caption"],
        "status": meta["status"],
        "image_png_base64": image_b64,
    }
    for optional in ("note", "group"):
        if meta.get(optional):
            out[optional] = meta[optional]
    out.update(extra)
    return out


# --------------------------------------------------------------------------
# Encoding helpers
# --------------------------------------------------------------------------

def _png_b64(array_or_img) -> str:
    """uint8 array (H,W) or (H,W,3), or a PIL image -> base64 PNG."""
    img = array_or_img
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = _fit(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fit(img: Image.Image) -> Image.Image:
    """Downscale so the long edge is at most VIEW_SIZE. Never upscales."""
    w, h = img.size
    scale = VIEW_SIZE / max(w, h)
    if scale >= 1.0:
        return img
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)


# --------------------------------------------------------------------------
# The segmenter, loaded once and only if it is actually there
# --------------------------------------------------------------------------

def _find_unet_checkpoint() -> Path | None:
    """The backend is started from serving/ in development and from /app in the
    container, so the checkpoint sits at a different depth each time. Try the
    plausible places instead of hard-coding one."""
    env = os.getenv("SEG_CHECKPOINT_PATH")
    here = Path(__file__).resolve().parent
    candidates = [Path(env)] if env else []
    candidates += [
        here / "checkpoints" / "unet_best.pth",
        Path("checkpoints") / "unet_best.pth",
        here.parent / "checkpoints" / "unet_best.pth",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _import_unet():
    """`segmentation/` sits next to this file, in development and in the
    container alike.

    It used to live at the repo root, because it was research code. It is not
    that any more: the classifier does not use it, nothing in `rsna/` needs it
    at serving time, and the only thing that still asks for it is the card this
    module draws. So it moved in here, where its one remaining consumer is.

    The fallback covers being started from a working directory other than this
    one, which happens in development often enough to be worth two lines.
    """
    try:
        from segmentation.unet import UNet  # noqa: PLC0415
        return UNet
    except ImportError:
        here = str(Path(__file__).resolve().parent)
        if here not in sys.path:
            sys.path.insert(0, here)
        from segmentation.unet import UNet  # noqa: PLC0415
        return UNet


_seg_model = None
_seg_error: str | None = None


def load_segmenter() -> bool:
    """Load the U-Net once. Returns False if it is unavailable; in that case the
    mask and crop tiles are reported as skipped and the rest still works."""
    global _seg_model, _seg_error
    if _seg_model is not None:
        return True
    if _seg_error is not None:
        return False
    path = _find_unet_checkpoint()
    if path is None:
        _seg_error = "unet_best.pth not found"
        return False
    try:
        UNet = _import_unet()
        model = UNet(base_ch=32)
        model.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True))
        model.eval()
        _seg_model = model
        return True
    except Exception as exc:  # noqa: BLE001
        _seg_error = f"{type(exc).__name__}: {exc}"
        return False


def segmenter_status() -> dict:
    return {"available": _seg_model is not None, "error": _seg_error}


# --------------------------------------------------------------------------
# Mask cleaning
# --------------------------------------------------------------------------
# Same idea as segmentation/mask_refine.py: close, fill holes, keep the two
# largest components. The symmetry fill from that module is deliberately left
# out here - it exists to make the mask AREA class-independent for training, and
# for a single image on screen it would invent a lung that the segmenter did not
# find. Showing the honest output matters more than showing a tidy one.

_KEEP_FRAC = 0.05


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """Flood-fill from the border; whatever the flood cannot reach is a hole."""
    m = (binary.astype(np.uint8) * 255)
    h, w = m.shape
    flood = m.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return ((m | holes) > 0)


def clean_mask(binary: np.ndarray) -> np.ndarray:
    """Closing -> fill holes -> keep the (up to) two largest components."""
    m = (binary.astype(np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = _fill_holes(m)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 2:                                   # background + at most one blob
        return m.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1] + 1          # label ids, largest first
    largest = areas.max()
    keep = [c for c in order[:2] if stats[c, cv2.CC_STAT_AREA] >= _KEEP_FRAC * largest]
    return np.isin(lbl, keep)


@torch.no_grad()
def lung_mask(pil_img: Image.Image) -> np.ndarray | None:
    """Binary lung mask at the resolution of the input image, or None.

    The segmenter gets exactly what it was trained on: greyscale, 256x256,
    values in [0,1] - NOT the classifier transform (no CLAHE, no
    standardisation). Feeding it the classifier input is the classic way to get
    a mask that looks almost right and is subtly wrong everywhere.
    """
    if not load_segmenter():
        return None
    small = pil_img.convert("L").resize((SEG_SIZE, SEG_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0)[None, None]
    pred = (torch.sigmoid(_seg_model(x))[0, 0].numpy() > 0.5)
    clean = clean_mask(pred)
    if not clean.any():
        return None
    w, h = pil_img.size
    full = cv2.resize(clean.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return full.astype(bool)


# --------------------------------------------------------------------------
# Crop geometry
# --------------------------------------------------------------------------

def square_crop_box(mask: np.ndarray, pad: float = CROP_PAD) -> tuple[int, int, int]:
    """(left, top, side) in pixels: the squared bounding box of the mask.

    Square rather than free-form on purpose. Scaling a non-square rectangle to
    224x224 distorts the image by an amount that depends on its aspect ratio, so
    the aspect ratio - itself a projection proxy - would survive the crop as a
    distortion signature (`rsna/befunde/rsna_crop_geometry.py`).

    A square that hangs over the edge is SHIFTED inwards, not shrunk. Shrinking
    would tie the crop size to the position in the image, and position is
    another projection proxy - the same leak through the back door.
    """
    ys, xs = np.where(mask)
    H, W = mask.shape
    top, bottom = ys.min(), ys.max() + 1
    left, right = xs.min(), xs.max() + 1

    # Margin relative to the respective side, applied BEFORE squaring: a single
    # margin added after squaring would scale with the longer side, and which
    # side is longer differs between upright and supine films.
    bh, bw = bottom - top, right - left
    top -= pad * bh
    bottom += pad * bh
    left -= pad * bw
    right += pad * bw

    side = min(max(bottom - top, right - left), H, W)
    cy, cx = (top + bottom) / 2.0, (left + right) / 2.0
    y0 = min(max(cy - side / 2.0, 0), H - side)
    x0 = min(max(cx - side / 2.0, 0), W - side)
    return int(round(x0)), int(round(y0)), int(round(side))


# --------------------------------------------------------------------------
# The individual stage pictures
# --------------------------------------------------------------------------

def render_original(pil_img: Image.Image) -> str:
    return _png_b64(pil_img.convert("RGB"))


def render_mask_overlay(pil_img: Image.Image, mask: np.ndarray) -> str:
    """Greyscale image with the mask laid over it in red, plus its outline."""
    base = np.asarray(pil_img.convert("L"))
    rgb = np.stack([base] * 3, axis=-1).astype(np.float32)
    tint = np.array([224.0, 87.0, 75.0])                     # --pos from styles.css
    m = mask[..., None]
    rgb = np.where(m, 0.62 * rgb + 0.38 * tint, rgb)
    rgb = rgb.astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    thickness = max(1, round(min(rgb.shape[:2]) / 200))
    cv2.drawContours(rgb, contours, -1, (255, 210, 205), thickness)
    return _png_b64(rgb)


# Kept, but no longer called by the running app: the crop tile was taken out of
# the interface because it read as a step of the scored path when it never was
# one. The code stays because it is what the crop findings were measured with
# and `test_*` still covers it. Nothing in serving/ calls it.
def render_crop(pil_img: Image.Image, mask: np.ndarray) -> tuple[str, dict]:
    """The cropped image plus the box it came from, in fractions of the original."""
    x0, y0, side = square_crop_box(mask)
    crop = pil_img.convert("RGB").crop((x0, y0, x0 + side, y0 + side))
    W, H = pil_img.size
    box = {
        "left": round(x0 / W, 4),
        "top": round(y0 / H, 4),
        "side_px": side,
        "kept_fraction": round((side * side) / float(W * H), 4),
    }
    return _png_b64(crop), box


def render_head_field(pil_img: Image.Image, field: np.ndarray) -> str:
    """Das 14x14-Kopffeld als weicher Verlauf ueber dem Bild.

    Drei Entscheidungen, und jede einzelne ist eine Aussage darueber, was die
    Messung traegt:

    KEINE SCHWELLE. Der Pegel des Kopfes ist unkalibriert: auf Bildern ohne
    Pneumonie schlaegt er in 62 Prozent der Faelle an (Phase 5b). Ein Kasten,
    eine Umrandung oder ein Schnitt bei 0,5 waere eine Behauptung ueber genau
    die Groesse, die nachgemessen und fuer untauglich befunden wurde.

    KEINE STRECKUNG JE BILD. Die Werte sind Wahrscheinlichkeiten zwischen 0 und
    1 und werden genau so eingefaerbt. Wuerde jedes Bild auf sein eigenes
    Maximum gestreckt, saehe ein Feld ohne jeden Ausschlag genauso deutlich aus
    wie ein starkes, und der Unterschied zwischen "hier" und "nirgends" waere
    weggerechnet.

    DECKKRAFT PROPORTIONAL ZUM WERT. Wo der Kopf nichts sagt, bleibt das
    Roentgenbild sichtbar. Das ist der bildliche Weg, "keine Aussage" von
    "Aussage: nein" zu unterscheiden.

    Bilineare Vergroesserung von 14 auf 224 und danach Begrenzung auf 0 bis 1:
    eine kubische Vergroesserung ueberschwingt und erzeugt Werte, die im Feld
    nicht stehen.
    """
    base = np.asarray(pil_img.convert("L").resize((224, 224), Image.BILINEAR))
    rgb = np.stack([base] * 3, axis=-1).astype(np.float32)

    f = cv2.resize(field.astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)
    f = np.clip(f, 0.0, 1.0)

    # INFERNO und nicht JET: der Grad-CAM daneben ist JET, und zwei Karten mit
    # derselben Farbskala werden fuer dieselbe Karte gehalten.
    colour = cv2.applyColorMap((f * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB).astype(np.float32)

    alpha = (0.75 * f)[..., None]
    out = (1.0 - alpha) * rgb + alpha * colour
    return _png_b64(np.clip(out, 0, 255).astype(np.uint8))


def render_head_field_layer(field: np.ndarray) -> str:
    """Dasselbe Kopffeld als DURCHSICHTIGE Ebene, ohne Roentgenbild darunter.

    Warum eine zweite Fassung und nicht nur die verrechnete: in der
    Ergebniskarte liegt ueber dem Bild ein Regler, und ein bereits verrechnetes
    Bild ueber demselben Bild einzublenden waere ein zweiter Durchgang derselben
    Mischung. Der Browser soll genau einmal mischen, mit genau dem Alpha, das
    hier steht.

    Die drei Entscheidungen aus `render_head_field` gelten unveraendert weiter,
    und sie sind hier sogar der ganze Inhalt der Datei: KEINE Schwelle, KEINE
    Streckung je Bild, DECKKRAFT PROPORTIONAL ZUM WERT. Wo der Kopf nichts sagt,
    ist die Ebene durchsichtig, und das Roentgenbild darunter bleibt
    unangetastet sichtbar. Genau das ist der bildliche Unterschied zwischen
    "keine Aussage" und "Aussage: nein".

    Der Regler in der Oberflaeche multipliziert dieses Alpha; bei Vollausschlag
    liegt es also bei 0,75 an der staerksten Stelle und nirgends darueber.
    """
    f = cv2.resize(field.astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)
    f = np.clip(f, 0.0, 1.0)

    colour = cv2.applyColorMap((f * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)

    alpha = (0.75 * f * 255).astype(np.uint8)
    rgba = np.dstack([colour, alpha])
    # Ueber PIL im RGBA-Modus, nicht ueber _png_b64: das dortige
    # Image.fromarray traefe zwar auch RGBA, aber _fit skaliert bilinear, und
    # 224 Pixel sind ohnehin unter VIEW_SIZE. Der direkte Weg spart die Frage.
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_model_input(tensor: torch.Tensor) -> str:
    """The normalised 224x224 tensor, re-stretched to 0-255 for display.

    After the ImageNet normalisation the values have no fixed range, so they
    cannot be shown directly. The stretch is a display device only - it is
    stated as such in the caption, because a picture that silently differs from
    the model input is worse than no picture.
    """
    t = tensor.detach()
    if t.ndim == 4:
        t = t[0]                       # (1,3,H,W) -> (3,H,W)
    arr = t[0].numpy()                 # the three channels are identical greyscale
    lo, hi = float(arr.min()), float(arr.max())
    norm = (arr - lo) / (hi - lo + 1e-6)
    return _png_b64((norm * 255).astype(np.uint8))
