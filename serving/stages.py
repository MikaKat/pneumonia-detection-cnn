"""Preprocessing stage images for the web app.

The web app shows the pipeline as a chain: the uploaded image, then one tile per
processing step, each connected by an arrow, ending in the Grad-CAM heatmap.
This module produces the picture for each tile. `main.py` calls the steps one by
one and publishes each result as soon as it exists, so the chain fills up in the
browser while the analysis is still running.

Honesty about what is shown
---------------------------
Two of the tiles are development stages, not production stages. The deployed
classifier runs on the FULL image, exactly as it was trained. The lung mask and
the lung crop were built and measured in this project (`segmentation/`,
`rsna/pipeline/rsna_make_crops.py`) and both were rejected on the RSNA data:
the pixel-exact mask destroys pathology and re-encodes the shape as a shortcut,
and the crop did not move the confounder score in the intended direction.
They are computed here for real and shown for real, but flagged as `explored`
so nobody reads the chain as "this is what the score came out of". `main.py`
sends that flag along; the frontend greys those tiles and states it in words.

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
# The model input stage is described differently depending on which weights the
# server loaded, because the two training runs preprocess differently. Kept next
# to each other so the difference is visible in one screen rather than hidden in
# an if. MODEL_FAMILY is set in main.py and checked there against the checkpoint.
_FAMILY = os.getenv("MODEL_FAMILY", "rsna").lower()

_FAMILY_TEXT = {
    "rsna": {
        "caption": "Resize to 224x224, greyscale, then the fixed ImageNet "
                   "normalisation the model was trained with. Computed on the full "
                   "uploaded image.",
        "note": "The normalisation subtracts the same constants from every image, so "
                "unlike the picture above it, brightness differences between images "
                "survive it. It is invisible here anyway: the tensor is re-stretched "
                "to 0-255 for display, and normalisation and stretch are both affine, "
                "so they cancel. What you can see is the resize. The model works on "
                "the unbounded values.",
    },
    "kermany": {
        "caption": "Resize to 224x224, CLAHE for local structure, then per-image "
                   "standardisation to mean 0 / std 1. Computed on the full "
                   "uploaded image.",
        "note": "The per-image standardisation is what removed the global "
                "brightness/contrast shortcut. It is invisible in this picture by "
                "construction: the tensor is re-stretched to 0-255 for display, and "
                "standardisation and stretch are both affine, so they cancel. What "
                "you can see here is the resize and CLAHE. The model works on the "
                "unbounded values.",
    },
}
if _FAMILY not in _FAMILY_TEXT:
    _FAMILY = "rsna"


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
        # Caption and note depend on which weights are loaded, because the two
        # training runs normalise differently and a caption describing the other
        # one would be a plain untruth. See MODEL_FAMILY in main.py.
        "title": "Resize and normalise",
        "caption": _FAMILY_TEXT[_FAMILY]["caption"],
        "status": "active",
        "note": _FAMILY_TEXT[_FAMILY]["note"],
    },
    {
        "key": "heatmap",
        "title": "Grad-CAM heatmap",
        "caption": "Where the last convolutional block contributed most to the score.",
        "status": "active",
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
    """`segmentation/` lives at the repo root in development and is copied next
    to this file in the container. Cover both without duplicating the model."""
    try:
        from segmentation.unet import UNet  # noqa: PLC0415
        return UNet
    except ImportError:
        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
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


def render_model_input(tensor: torch.Tensor) -> str:
    """The standardised 224x224 tensor, re-stretched to 0-255 for display.

    After PerImageStandardize the values are centred on 0 with std 1 and have no
    fixed range, so they cannot be shown directly. The stretch is a display
    device only - it is stated as such in the caption, because a picture that
    silently differs from the model input is worse than no picture.
    """
    t = tensor.detach()
    if t.ndim == 4:
        t = t[0]                       # (1,3,H,W) -> (3,H,W)
    arr = t[0].numpy()                 # the three channels are identical greyscale
    lo, hi = float(arr.min()), float(arr.max())
    norm = (arr - lo) / (hi - lo + 1e-6)
    return _png_b64((norm * 255).astype(np.uint8))
