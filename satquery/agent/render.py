"""Turning arrays into the pictures a human looks at: previews and overlays.

Two consumers, one transform. The preview in the GUI and the image handed to the
VLM go through *the same* display pipeline -- percentile stretch for optical,
dB clip for SAR -- because a reviewer comparing "what the model saw" against
"what I saw" must not be looking at two different renderings. That is why this
module reuses :mod:`satquery.preprocess` rather than doing its own scaling.

Overlays are drawn on top of that rendering, never on raw data:

* a change mask goes down as a translucent colour wash plus a hard outline, so a
  thin linear change (a new road) is still visible at thumbnail size;
* grounding boxes are drawn with their score, because a box with no score is an
  assertion and a box with 0.31 on it is evidence.

PIL is required here and only here. It ships in the ``data`` extra, which every
environment that can display anything already has; a missing PIL degrades the
GUI, not the numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..contracts import Box, SatQueryError
from ..io.raster import RasterImage
from ..log import get_logger
from ..preprocess.optical import percentile_stretch, to_uint8
from ..preprocess.sar import db_to_uint8, linear_to_db
from ..preprocess.tiling import resize_max_side

__all__ = [
    "display_array",
    "save_png",
    "render_preview",
    "overlay_mask",
    "draw_boxes",
    "MASK_COLOR",
]

_LOG = get_logger(__name__)

#: Change-mask wash. Magenta because it collides with nothing in an RS scene --
#: vegetation is green, water blue-black, built-up grey, bare soil brown.
MASK_COLOR: tuple[int, int, int] = (255, 0, 170)

_BOX_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 215, 0),
    (0, 229, 255),
    (124, 252, 0),
    (255, 105, 180),
    (255, 140, 0),
)


def _pil() -> Any:
    try:
        from PIL import Image  # noqa: F401
        from PIL import ImageDraw  # noqa: F401

        import PIL

        return PIL
    except ImportError as exc:  # pragma: no cover - pillow is in the data extra
        raise SatQueryError(
            'rendering needs Pillow: pip install "satquery[data]"'
        ) from exc


def display_array(
    image: RasterImage | np.ndarray,
    *,
    modality: str = "optical",
    max_side: int = 768,
) -> np.ndarray:
    """Render to ``(3, h, w)`` uint8 exactly the way the VLM is fed.

    The order is stretch -> resize -> quantise, matching
    :func:`satquery.models.qwen_vl.prepare_image`: quantising last means the
    8-bit levels are computed once, at the final resolution, instead of being
    resampled after rounding.
    """
    array = np.asarray(getattr(image, "array", image), dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise SatQueryError(f"expected (h, w) or (band, h, w), got shape {array.shape}")

    if modality == "sar":
        array = np.asarray(linear_to_db(array), dtype=np.float32)
        array = np.asarray(resize_max_side(array, max_side=max_side), dtype=np.float32)
        eight_bit = np.asarray(db_to_uint8(array), dtype=np.uint8)
    else:
        array = np.asarray(percentile_stretch(array, low=2.0, high=98.0), dtype=np.float32)
        array = np.asarray(resize_max_side(array, max_side=max_side), dtype=np.float32)
        eight_bit = np.asarray(to_uint8(array), dtype=np.uint8)

    return _as_rgb(eight_bit, modality)


def _as_rgb(array: np.ndarray, modality: str) -> np.ndarray:
    """Three bands out of however many came in, matching the predictor's choice."""
    bands = int(array.shape[0])
    if bands >= 3:
        return np.ascontiguousarray(array[:3])
    if bands == 2:  # SAR VV/VH false colour: R=VV, G=VH, B=VV
        return np.ascontiguousarray(array[[0, 1, 0]])
    return np.ascontiguousarray(np.repeat(array[:1], 3, axis=0))


def save_png(array: np.ndarray, path: str | Path) -> Path:
    """Write a ``(3, h, w)`` / ``(h, w)`` uint8 array as PNG. Returns the path."""
    pil = _pil()
    a = np.asarray(array)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        image = pil.Image.fromarray(a, mode="L")
    elif a.ndim == 3 and a.shape[0] in (1, 3, 4):
        hwc = np.ascontiguousarray(np.transpose(a, (1, 2, 0)))
        image = pil.Image.fromarray(hwc[:, :, 0], mode="L") if hwc.shape[2] == 1 else pil.Image.fromarray(hwc)
    else:
        raise SatQueryError(f"cannot save an array of shape {a.shape} as PNG")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    image.save(p)
    return p


def render_preview(
    image: RasterImage | np.ndarray,
    path: str | Path,
    *,
    modality: str = "optical",
    max_side: int = 768,
) -> Path:
    """Display-transform an image and write it as a PNG for the GUI."""
    return save_png(display_array(image, modality=modality, max_side=max_side), path)


def overlay_mask(
    base: np.ndarray,
    mask: np.ndarray,
    *,
    color: Sequence[int] = MASK_COLOR,
    alpha: float = 0.45,
    outline: bool = True,
) -> np.ndarray:
    """Blend a boolean mask over a ``(3, h, w)`` uint8 rendering.

    The mask is resampled to the base's grid with nearest-neighbour when the
    shapes differ -- bilinear would invent fractional class membership at every
    boundary, and the boundary is the whole content of a change mask.
    """
    rgb = np.array(base, dtype=np.float32, copy=True)
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise SatQueryError(f"base must be (3, h, w) uint8, got shape {rgb.shape}")
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[0]
    m = m.astype(bool)
    if m.shape != rgb.shape[1:]:
        from ..preprocess.tiling import resample_to

        m = resample_to(m.astype(np.uint8), height=rgb.shape[1], width=rgb.shape[2], order=0).astype(bool)

    a = float(np.clip(alpha, 0.0, 1.0))
    for channel in range(3):
        rgb[channel][m] = (1.0 - a) * rgb[channel][m] + a * float(color[channel])

    if outline and m.any():
        edge = _edges(m)
        for channel in range(3):
            rgb[channel][edge] = float(color[channel])
    return np.clip(rgb, 0, 255).astype(np.uint8)


def draw_boxes(
    base: np.ndarray,
    boxes: Iterable[Box],
    *,
    width: int = 3,
    show_scores: bool = True,
) -> np.ndarray:
    """Draw grounding boxes with labels and scores on a ``(3, h, w)`` uint8 image.

    Boxes are in absolute pixels of the *rendered* image, which is the contract
    in :mod:`satquery.contracts`. A caller that grounded on a full-resolution
    scene and renders a 768 px preview must scale them first -- doing it here
    would need a scale factor this function has no way to know.
    """
    pil = _pil()
    a = np.asarray(base, dtype=np.uint8)
    hwc = np.ascontiguousarray(np.transpose(a, (1, 2, 0)))
    image = pil.Image.fromarray(hwc)
    draw = pil.ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        color = _BOX_COLORS[i % len(_BOX_COLORS)]
        draw.rectangle([box.x0, box.y0, box.x1, box.y1], outline=color, width=int(width))
        label = box.label or "match"
        if show_scores and box.score is not None:
            label = f"{label} {box.score:.2f}"
        y = max(0.0, box.y0 - 12.0)
        draw.text((box.x0 + 2, y), label, fill=color)
    return np.ascontiguousarray(np.transpose(np.asarray(image, dtype=np.uint8), (2, 0, 1)))


def _edges(mask: np.ndarray) -> np.ndarray:
    """One-pixel boundary of a boolean mask, by four-neighbour comparison."""
    padded = np.pad(mask, 1, mode="edge")
    neighbours = (
        padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    return mask & ~neighbours
