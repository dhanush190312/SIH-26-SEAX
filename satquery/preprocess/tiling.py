"""Tiling, stitching and the 768 px resize every VLM call goes through.

Two jobs that look unrelated but are the same problem -- a satellite scene is
the wrong size for a vision transformer -- and the project resolves it in two
directions:

* **Down**: :func:`resize_max_side` caps the longest side at 768 px, the
  training cap for the Qwen2.5-VL LoRA run. It must give identical results on
  the laptop and on Colab, so it is implemented in numpy alone rather than
  deferring to whichever PIL/scipy happens to be installed. Every single-image
  VLM call passes through here, which makes its resampling convention
  (half-pixel centres, see :func:`_triangle_weights`) one that must never drift.
* **Across**: :func:`tile_image` cuts a large scene into 512x512 windows with
  64 px overlap so a 10k x 10k Cartosat strip can be answered tile by tile, and
  :func:`merge_tiles` puts per-tile outputs (masks, index maps, change maps)
  back into scene geometry. The overlap exists because a model's answer degrades
  near a tile edge where context is cut off; averaging the overlap is the
  cheapest way to stop tile seams appearing as fake linear features in a mask,
  which on RS imagery reads as a road.

The round-trip property is load-bearing and tested both ways: tiling an array
and merging it with ``reduce="mean"`` returns the original, for scene sizes that
are and are not multiples of the stride. Anything that breaks it silently shifts
every mask by a few pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from ..log import get_logger

__all__ = [
    "DEFAULT_TILE_SIZE",
    "DEFAULT_TILE_OVERLAP",
    "DEFAULT_MAX_SIDE",
    "REDUCERS",
    "Tile",
    "tile_image",
    "merge_tiles",
    "resize_max_side",
    "resample_to",
]

_LOG = get_logger(__name__)

#: Inference tiling geometry from the approach doc: 512 px windows, 64 px
#: overlap. The overlap is a margin wide enough that the degraded band at a
#: tile edge is always covered by a neighbour, not a tuned number -- raise it
#: if seams show in merged masks.
DEFAULT_TILE_SIZE: int = 512
DEFAULT_TILE_OVERLAP: int = 64

#: Longest side handed to a VLM. Matches ``SatQueryConfig.max_side`` and the
#: fine-tuning recipe; changing one without the other means inference sees a
#: resolution the adapter never trained on.
DEFAULT_MAX_SIDE: int = 768

Reducer = Literal["mean", "max", "min", "first", "last"]

#: How overlapping tiles are combined by :func:`merge_tiles`. ``mean`` for
#: continuous output (probabilities, index maps, imagery), ``max`` for detection
#: heat where a hit anywhere should win, ``first``/``last`` when the tiles are
#: known to agree and the cheap path is enough.
REDUCERS: tuple[str, ...] = ("mean", "max", "min", "first", "last")


# eq=False on purpose: the default frozen-dataclass ``__eq__``/``__hash__``
# compare and hash every field, so the first ``tile_a == tile_b`` or ``set()``
# of tiles would raise "truth value of an array is ambiguous". Identity
# semantics are predictable; compare ``.array`` explicitly to compare pixels.
@dataclass(frozen=True, slots=True, eq=False)
class Tile:
    """One window of a scene, plus where it came from.

    ``y0``/``x0`` are the window's top-left corner in *scene* pixel
    coordinates. With ``pad=True`` a bottom or right tile can run past the scene
    (``y0 + height > scene height``); those columns carry replicated edge values
    and :func:`merge_tiles` drops them.

    Keep the object rather than re-deriving offsets downstream: a box predicted
    in tile coordinates becomes a scene box by adding ``(x0, y0)``, and getting
    that offset wrong is the classic tiled-grounding bug -- every box lands in
    the first tile.
    """

    array: np.ndarray
    row: int
    col: int
    y0: int
    x0: int

    @property
    def height(self) -> int:
        return int(self.array.shape[-2])

    @property
    def width(self) -> int:
        return int(self.array.shape[-1])

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Scene-space ``(x0, y0, x1, y1)``, in :class:`~satquery.contracts.Box` order."""
        return (self.x0, self.y0, self.x0 + self.width, self.y0 + self.height)


def tile_image(
    array: np.ndarray,
    *,
    size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_TILE_OVERLAP,
    pad: bool = True,
) -> list[Tile]:
    """Cut ``(band, h, w)`` or ``(h, w)`` into overlapping windows, row-major.

    Stride is ``size - overlap``. Start offsets march from the top-left until
    the scene is covered, so the last row and column are where the two modes
    differ, and the choice matters:

    * ``pad=True`` (default) keeps every tile exactly ``size x size`` by
      replicating edge pixels past the scene boundary. Batched inference needs
      uniform shapes, and edge replication -- not zeros -- because a black
      border is a strong, entirely artificial edge that a model happily reports
      as a coastline or a field boundary.
    * ``pad=False`` slides the last window back so it ends flush with the scene.
      No invented pixels, at the cost of a larger overlap on that row/column. A
      scene smaller than ``size`` yields one short tile instead of a padded one.

    Either way the tiles cover the scene completely and :func:`merge_tiles`
    reconstructs it exactly. Tile arrays are contiguous copies, so a caller may
    normalise or overwrite a tile in place without corrupting the scene.
    """
    a = np.asarray(array)
    if a.ndim not in (2, 3):
        raise ValueError(f"expected (h, w) or (band, h, w), got shape {a.shape}")
    if int(size) != size or size < 1:
        raise ValueError(f"size must be an integer >= 1, got {size!r}")
    size = int(size)
    if int(overlap) != overlap or overlap < 0:
        raise ValueError(f"overlap must be an integer >= 0, got {overlap!r}")
    overlap = int(overlap)
    if overlap >= size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than size ({size}): stride would be <= 0"
        )
    height, width = int(a.shape[-2]), int(a.shape[-1])
    if height == 0 or width == 0:
        raise ValueError(f"cannot tile an empty array, got shape {a.shape}")

    stride = size - overlap
    ys = _tile_starts(height, size, stride, pad)
    xs = _tile_starts(width, size, stride, pad)

    if pad:
        pad_y = max(0, ys[-1] + size - height)
        pad_x = max(0, xs[-1] + size - width)
        if pad_y or pad_x:
            widths = [(0, 0)] * (a.ndim - 2) + [(0, pad_y), (0, pad_x)]
            a = np.pad(a, widths, mode="edge")

    tiles: list[Tile] = []
    for row, y0 in enumerate(ys):
        for col, x0 in enumerate(xs):
            window = a[..., y0 : y0 + size, x0 : x0 + size]
            tiles.append(Tile(array=np.ascontiguousarray(window), row=row, col=col, y0=y0, x0=x0))
    return tiles


def merge_tiles(
    tiles: Sequence[Tile],
    *,
    height: int,
    width: int,
    reduce: str = "mean",
) -> np.ndarray:
    """Stitch tiles back into a ``(band, height, width)`` / ``(height, width)`` scene.

    ``height``/``width`` are the *scene* size and are required rather than
    inferred: padded tiles in the last row extend past it, so guessing the
    extent from the tiles would return an array slightly larger than the
    georeferencing says the scene is, and every subsequent pixel-to-coordinate
    conversion would be wrong at the bottom edge.

    ``reduce="mean"`` is the one that round-trips. It averages only the tiles
    that actually cover a pixel, so the interior is not pulled toward the
    zero-initialised accumulator -- that bug shows up as a regular grid of dark
    seams, which is exactly the artefact the overlap exists to avoid. Pixels no
    tile covers come back as 0 and are logged; that only happens when the caller
    passes a subset of the tiles.

    Integer and boolean tiles are accumulated in float64 and rounded back to the
    input dtype, so a merged uint8 mosaic stays uint8 instead of silently
    quadrupling the memory of a large scene.
    """
    if reduce not in REDUCERS:
        raise ValueError(f"unknown reduce {reduce!r}, expected one of {REDUCERS}")
    if not tiles:
        raise ValueError("merge_tiles needs at least one tile")
    if height < 1 or width < 1:
        raise ValueError(f"scene size must be positive, got height={height}, width={width}")

    first = np.asarray(tiles[0].array)
    if first.ndim not in (2, 3):
        raise ValueError(f"tile arrays must be (h, w) or (band, h, w), got shape {first.shape}")
    bands = int(first.shape[0]) if first.ndim == 3 else 0
    dtype = first.dtype

    shape = (bands, height, width) if bands else (height, width)
    fill = {"max": -np.inf, "min": np.inf}.get(reduce, 0.0)
    acc = np.full(shape, fill, dtype=np.float64)
    count = np.zeros((height, width), dtype=np.int64)

    for tile in tiles:
        t = np.asarray(tile.array)
        if t.ndim != first.ndim or (bands and int(t.shape[0]) != bands):
            raise ValueError(
                f"tile (row={tile.row}, col={tile.col}) has shape {t.shape}, incompatible "
                f"with the first tile's {first.shape}"
            )
        y0, x0 = int(tile.y0), int(tile.x0)
        sy0, sx0 = max(y0, 0), max(x0, 0)
        sy1, sx1 = min(y0 + tile.height, height), min(x0 + tile.width, width)
        if sy1 <= sy0 or sx1 <= sx0:
            continue  # wholly outside the scene: an all-pad tile of a subset
        src = t[..., sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0].astype(np.float64, copy=False)
        dst = (..., slice(sy0, sy1), slice(sx0, sx1))
        if reduce == "mean":
            acc[dst] += src
        elif reduce == "max":
            acc[dst] = np.maximum(acc[dst], src)
        elif reduce == "min":
            acc[dst] = np.minimum(acc[dst], src)
        elif reduce == "last":
            acc[dst] = src
        else:  # "first": keep whatever an earlier tile already wrote
            seen = count[sy0:sy1, sx0:sx1] > 0
            acc[dst] = np.where(seen, acc[dst], src)
        count[sy0:sy1, sx0:sx1] += 1

    covered = count > 0
    if not covered.all():
        _LOG.warning(
            "merge_tiles: %d of %d scene pixels covered by no tile, filled with 0",
            int((~covered).sum()),
            height * width,
        )
    plane = covered[None, ...] if bands else covered
    if reduce == "mean":
        divisor = np.where(covered, count, 1).astype(np.float64)
        acc = acc / (divisor[None, ...] if bands else divisor)
    acc = np.where(plane, acc, 0.0)
    return _cast_like(acc, dtype)


def resample_to(
    array: np.ndarray,
    *,
    height: int,
    width: int,
    order: int = 1,
) -> np.ndarray:
    """Resample a ``(band, h, w)`` / ``(h, w)`` array onto an exact pixel grid.

    :func:`resize_max_side` caps a scene while preserving its aspect ratio; this
    is the other half, used when a *target grid* is given rather than a budget
    -- putting a SAR image onto an optical image's grid before fusion, or a
    change mask back onto the original scene size. Aspect ratio is not
    preserved, because the caller asking for a specific grid already decided
    that.

    ``order=0`` (nearest) for anything categorical. Same numpy implementation as
    :func:`resize_max_side`, so laptop and Colab agree bit for bit.
    """
    if order not in (0, 1):
        raise ValueError(f"order must be 0 (nearest) or 1 (bilinear), got {order!r}")
    if int(height) != height or int(width) != width or height < 1 or width < 1:
        raise ValueError(f"height and width must be integers >= 1, got {(height, width)!r}")
    a = np.asarray(array)
    if a.ndim not in (2, 3):
        raise ValueError(f"expected (h, w) or (band, h, w), got shape {a.shape}")
    if a.shape[-2] == 0 or a.shape[-1] == 0:
        raise ValueError(f"cannot resample an empty array, got shape {a.shape}")
    if (int(a.shape[-2]), int(a.shape[-1])) == (int(height), int(width)):
        return a
    out = a.astype(np.float64, copy=False)
    out = _resample_axis(out, axis=out.ndim - 2, dst_len=int(height), order=order)
    out = _resample_axis(out, axis=out.ndim - 1, dst_len=int(width), order=order)
    return _cast_like(out, a.dtype)


def resize_max_side(
    array: np.ndarray,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
    order: int = 1,
) -> np.ndarray:
    """Scale a ``(band, h, w)`` / ``(h, w)`` array so its longest side is ``max_side``.

    A cap, not a resize: an image already within ``max_side`` comes back
    untouched (the same object, not a copy), because upsampling an RS tile
    invents detail and costs the VLM vision tokens for nothing.

    ``order=1`` is bilinear, ``order=0`` nearest. Use 0 for anything categorical
    -- label masks, change maps, class indices -- where interpolating between
    class 1 and class 3 fabricates class 2.

    The bilinear path widens its triangle filter when downscaling, i.e. it
    area-averages rather than point-samples. That is what PIL and
    ``skimage.transform.resize(anti_aliasing=True)`` do, and it matters more
    here than in ordinary photography: SAR speckle and single-pixel targets are
    high-frequency by nature, so plain sampling at a 5x reduction turns a
    speckled field into moire and can drop a thin road entirely. Implemented in
    numpy so a laptop baseline and a Colab baseline agree; NaN is not handled
    specially and spreads over its filter footprint, so fill nodata first.
    """
    if order not in (0, 1):
        raise ValueError(f"order must be 0 (nearest) or 1 (bilinear), got {order!r}")
    if int(max_side) != max_side or max_side < 1:
        raise ValueError(f"max_side must be an integer >= 1, got {max_side!r}")
    max_side = int(max_side)
    a = np.asarray(array)
    if a.ndim not in (2, 3):
        raise ValueError(f"expected (h, w) or (band, h, w), got shape {a.shape}")
    height, width = int(a.shape[-2]), int(a.shape[-1])
    if height == 0 or width == 0:
        raise ValueError(f"cannot resize an empty array, got shape {a.shape}")
    if max(height, width) <= max_side:
        return a

    # Pin the long side exactly on max_side and derive the short one, instead of
    # scaling both by a float factor: independent rounding can leave the long
    # side at 767 and break a caller that asserts the cap was applied.
    if height >= width:
        new_h, new_w = max_side, max(1, _round_half_up(width * max_side / height))
    else:
        new_h, new_w = max(1, _round_half_up(height * max_side / width)), max_side

    out = a.astype(np.float64, copy=False)
    # Rows first: on a downscale that shrinks the array before the second pass
    # walks it, which is the cheaper order.
    out = _resample_axis(out, axis=out.ndim - 2, dst_len=new_h, order=order)
    out = _resample_axis(out, axis=out.ndim - 1, dst_len=new_w, order=order)
    return _cast_like(out, a.dtype)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _tile_starts(length: int, size: int, stride: int, pad: bool) -> tuple[int, ...]:
    """Window start offsets covering ``[0, length)``, in order.

    The loop condition is "the previous window did not reach the end", which
    keeps a scene of exactly ``size`` at one tile and a scene one pixel longer
    at two. ``range(0, length, stride)`` gets that wrong in the other direction:
    it emits a final window whose pixels are all already covered.
    """
    if length <= size:
        return (0,)
    starts = [0]
    while starts[-1] + size < length:
        starts.append(starts[-1] + stride)
    if not pad:
        # Pull the last window back inside the scene. It cannot collide with the
        # previous start: the loop appended only because ``starts[-2] + size <
        # length``, i.e. ``starts[-2] < length - size``.
        starts[-1] = min(starts[-1], length - size)
    return tuple(starts)


def _round_half_up(value: float) -> int:
    """Round half away from zero, for positive values.

    ``round()`` is banker's rounding, so ``round(0.5) == 0`` while ``round(1.5)
    == 2``: an output size would depend on parity, which is a surprising way for
    two near-identical scenes to end up one pixel apart.
    """
    return int(math.floor(value + 0.5))


def _resample_axis(a: np.ndarray, *, axis: int, dst_len: int, order: int) -> np.ndarray:
    """Resample one axis of a float64 array to ``dst_len``; other axes untouched."""
    src_len = int(a.shape[axis])
    if src_len == dst_len:
        return a
    if order == 0:
        idx = np.minimum(
            np.floor((np.arange(dst_len) + 0.5) * (src_len / dst_len)).astype(np.intp),
            src_len - 1,
        )
        return np.take(a, idx, axis=axis)
    idx, weights = _triangle_weights(src_len, dst_len)
    moved = np.moveaxis(a, axis, 0)
    out = np.zeros((dst_len,) + moved.shape[1:], dtype=np.float64)
    tail = (1,) * (moved.ndim - 1)
    # One gather per band of the filter footprint: the footprint is ~2 wide when
    # upscaling and ~2*src/dst when downscaling, so this stays a handful of
    # passes and never materialises a dense (dst, src) weight matrix.
    for j in range(idx.shape[1]):
        out += moved[idx[:, j]] * weights[:, j].reshape((dst_len,) + tail)
    return np.moveaxis(out, 0, axis)


def _triangle_weights(src_len: int, dst_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Banded ``(idx, weights)`` of shape ``(dst_len, k)`` for a 1-D triangle filter.

    Conventions, all of which are PIL's and none of which are arbitrary:

    * Half-pixel centres: output pixel ``i`` samples source coordinate
      ``(i + 0.5) * src/dst``. This is the only mapping that is symmetric under
      flipping the axis, so resizing an image and resizing its mirror give
      mirrored results -- ``align_corners``-style mapping does not, and the
      asymmetry shows up as a half-pixel shift in merged masks.
    * The support widens with ``src/dst`` when downscaling, so every source
      pixel contributes somewhere. Without it a 10x reduction reads one pixel in
      ten and discards the rest.
    * Weights are normalised per output pixel, which is what keeps a constant
      image constant at the border, where the support is clipped by the edge.
    """
    scale = src_len / dst_len
    filter_scale = max(1.0, scale)
    support = filter_scale  # the triangle's radius is 1 in filter space
    centers = (np.arange(dst_len, dtype=np.float64) + 0.5) * scale
    lo = np.maximum(np.floor(centers - support + 0.5), 0.0).astype(np.intp)
    hi = np.minimum(np.floor(centers + support + 0.5), float(src_len)).astype(np.intp)
    k = max(1, int(np.max(hi - lo)))
    idx = lo[:, None] + np.arange(k, dtype=np.intp)[None, :]
    inside = idx < hi[:, None]
    idx = np.clip(idx, 0, src_len - 1)
    t = (idx + 0.5 - centers[:, None]) / filter_scale
    weights = np.where(inside, np.maximum(1.0 - np.abs(t), 0.0), 0.0)
    total = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
    return idx, weights


def _cast_like(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Return a float64 result as ``dtype``, rounding and clipping for integers."""
    if dtype == np.bool_:
        return values >= 0.5
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(values), info.min, info.max).astype(dtype)
    return values.astype(dtype, copy=False)
