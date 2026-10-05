"""Generate the demo imagery: two bi-temporal optical scenes and an optical+SAR pair.

Synthetic, but not blocky. The demo has to *look* like remote sensing on a
screen recording, and a flat colour-block test fixture does not -- so this builds
a land-cover map (water, forest, fields, bare soil, built-up, roads), renders it
with correlated texture, and derives every band from the physics that band would
actually see:

* **RGB** -- reflectance per class plus smoothed noise, so fields have grain and
  a forest edge is not a straight line.
* **NIR** -- high over vegetation, near zero over water. This is what makes NDVI
  and NDWI real numbers in the demo rather than an RGB proxy.
* **SAR VV/VH** -- gamma speckle scaled per class: water dark (specular
  reflection away from the sensor), built-up bright (double bounce off walls),
  vegetation in between (volume scattering).

The *after* scene changes three things a viewer can see and the tools can
measure: a new development block, a cleared patch of forest, and a slightly
different sun angle -- the last one being the distractor that a naive image
difference reports as change everywhere.

Run: ``python tools/make_demo_scenes.py`` -> writes into ``demo/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SIZE = 512
SEED = 20260826

#: Class ids used in the land-cover map.
WATER, FOREST, FIELD_A, FIELD_B, BARE, URBAN, ROAD = range(7)

#: Per class: (red, green, blue, nir) reflectance, and a SAR backscatter scale.
#: The NIR column is the one that matters -- vegetation reflects it strongly and
#: water absorbs it, which is the entire basis of NDVI and NDWI.
SIGNATURES: dict[int, tuple[tuple[float, float, float, float], float]] = {
    WATER:   ((0.030, 0.052, 0.078, 0.020), 0.05),
    FOREST:  ((0.048, 0.082, 0.046, 0.400), 0.85),
    FIELD_A: ((0.115, 0.150, 0.088, 0.330), 0.55),
    FIELD_B: ((0.135, 0.168, 0.095, 0.300), 0.45),
    BARE:    ((0.290, 0.240, 0.175, 0.300), 0.35),
    URBAN:   ((0.215, 0.210, 0.205, 0.230), 3.20),
    ROAD:    ((0.145, 0.142, 0.145, 0.150), 1.40),
}

# FIELD_B is a second crop type, and its NIR is deliberately well clear of its
# green: at NIR ~0.215 the two sat close enough that texture noise pushed
# individual pixels across the NDWI = 0 water cut, and the grounding tool then
# boxed patches of dry crop as water. The threshold is right; the scene was
# ambiguous.

#: How much fine grain each class shows. Bare soil and ploughed fields are
#: textured at metre scale; open water is nearly featureless, and rendering it
#: with the same grain as a field is the single thing that makes synthetic
#: imagery read as a diagram.
GRAIN: dict[int, float] = {
    WATER: 0.05, FOREST: 0.34, FIELD_A: 0.20, FIELD_B: 0.24,
    BARE: 0.30, URBAN: 0.38, ROAD: 0.12,
}


def smooth(array: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur, scipy when present and a separable box cascade when not."""
    try:
        from scipy.ndimage import gaussian_filter

        return np.asarray(gaussian_filter(array, sigma=sigma))
    except ImportError:  # three box passes approximate a Gaussian well enough
        out = array.astype(np.float64)
        width = max(1, int(sigma * 2) | 1)
        kernel = np.ones(width) / width
        for axis in (0, 1):
            for _ in range(3):
                out = np.apply_along_axis(
                    lambda row: np.convolve(row, kernel, mode="same"), axis, out
                )
        return out


def field(rng: np.random.Generator, sigma: float, size: int = SIZE) -> np.ndarray:
    """A smooth random field in roughly [0, 1] -- the terrain-like base."""
    raw = rng.normal(0.0, 1.0, size=(size, size))
    out = smooth(raw, sigma)
    return (out - out.min()) / max(1e-9, out.ptp())


def land_cover(rng: np.random.Generator) -> np.ndarray:
    """The class map everything else is derived from."""
    cover = np.full((SIZE, SIZE), FIELD_A, dtype=np.int16)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]

    # Forest in the north-west, with a soft irregular boundary.
    forest = smooth(rng.normal(0, 1, (SIZE, SIZE)), 18.0)
    forest = (forest - forest.mean()) / forest.std()
    cover[(forest > 0.35) & (yy < SIZE * 0.62) & (xx < SIZE * 0.58)] = FOREST

    # A patchwork of fields in the south-east: rectangles on a jittered grid.
    for y0 in range(int(SIZE * 0.55), SIZE, 58):
        for x0 in range(int(SIZE * 0.45), SIZE, 66):
            h = 58 - int(rng.integers(6, 18))
            w = 66 - int(rng.integers(6, 20))
            cover[y0 : y0 + h, x0 : x0 + w] = FIELD_A if rng.random() < 0.5 else FIELD_B

    # A meandering river from the north edge to the south-east corner.
    t = np.linspace(0.0, 1.0, SIZE)
    centre = (
        SIZE * 0.22
        + SIZE * 0.30 * t
        + 26.0 * np.sin(t * 7.0)
        + 12.0 * np.sin(t * 17.0 + 1.3)
    )
    half_width = 5.0 + 3.5 * t
    cover[np.abs(xx - centre[yy]) < half_width[yy]] = WATER

    # A reservoir in the south-west.
    lake = ((yy - SIZE * 0.78) / (SIZE * 0.10)) ** 2 + ((xx - SIZE * 0.17) / (SIZE * 0.13)) ** 2
    cover[lake < 1.0] = WATER

    # Bare ground where the terrain is highest.
    elevation = field(rng, 26.0)
    cover[(elevation > 0.86) & (cover == FIELD_A)] = BARE

    _town(cover, rng, y0=int(SIZE * 0.14), x0=int(SIZE * 0.66), rows=4, cols=5)
    _road(cover, y=int(SIZE * 0.30), horizontal=True)
    _road(cover, y=int(SIZE * 0.62), horizontal=True)
    _road(cover, y=int(SIZE * 0.72), horizontal=False)
    return cover


def _town(cover: np.ndarray, rng: np.random.Generator, *, y0: int, x0: int, rows: int, cols: int) -> None:
    """A block of buildings on a street grid: the double-bounce SAR target."""
    for r in range(rows):
        for c in range(cols):
            if rng.random() < 0.15:
                continue
            top = y0 + r * 22 + int(rng.integers(0, 4))
            left = x0 + c * 24 + int(rng.integers(0, 4))
            cover[top : top + 14 + int(rng.integers(0, 5)),
                  left : left + 15 + int(rng.integers(0, 6))] = URBAN


def _road(cover: np.ndarray, *, y: int, horizontal: bool) -> None:
    if horizontal:
        cover[y : y + 4, :] = ROAD
    else:
        cover[:, y : y + 4] = ROAD


def render(cover: np.ndarray, rng: np.random.Generator, *, brightness: float = 1.0) -> np.ndarray:
    """``(4, h, w)`` float32 reflectance: R, G, B, NIR.

    Realism here is not vanity: a flat colour-block scene reads as a diagram on a
    screen recording, and a viewer cannot tell whether the tools are measuring
    imagery or reading their own fixture. Four things do the work:

    * **multi-scale texture** -- fine grain at ~1 px and slow variation at ~10 px,
      multiplied rather than added, so bright surfaces vary more than dark ones
      exactly as reflectance does;
    * **per-class grain** (:data:`GRAIN`) -- water is nearly featureless, ploughed
      soil is not;
    * **per-patch drift** -- a slow field per class, so two neighbouring fields of
      the same crop are not the same colour;
    * **a soft point-spread blur** at the end, because no sensor resolves a
      land-cover boundary as a hard pixel edge.
    """
    bands = np.zeros((4, SIZE, SIZE), dtype=np.float32)
    fine = field(rng, 1.1) - 0.5
    medium = field(rng, 9.0) - 0.5
    drift = field(rng, 34.0) - 0.5

    for class_id, (reflectance, _sar) in SIGNATURES.items():
        mask = cover == class_id
        if not mask.any():
            continue
        grain = GRAIN[class_id]
        modulation = (
            1.0
            + grain * 1.9 * fine[mask]
            + grain * 1.1 * medium[mask]
            + 0.34 * drift[mask]
        )
        for band, value in enumerate(reflectance):
            # Vegetation NIR varies with vigour rather than with the visible
            # grain, so the two are decorrelated -- which is what stops NDVI
            # from being a constant per class.
            extra = 0.55 * drift[mask] if band == 3 and class_id in (FOREST, FIELD_A, FIELD_B) else 0.0
            bands[band][mask] = value * (modulation + extra)

    # Roof-by-roof brightness: a town is not one grey.
    urban = cover == URBAN
    if urban.any():
        roofs = field(rng, 2.0)[urban]
        for band in range(4):
            bands[band][urban] *= 0.62 + 0.85 * roofs

    bands *= brightness
    bands = np.stack([smooth(band, 0.62) for band in bands]).astype(np.float32)
    bands += rng.normal(0.0, 0.0035, bands.shape).astype(np.float32)  # sensor noise
    # A touch of atmospheric haze: additive, strongest in blue. Without it the
    # shadows are unnaturally black for a top-of-atmosphere product.
    bands += np.array([0.012, 0.016, 0.028, 0.006], dtype=np.float32)[:, None, None]
    return np.clip(bands, 0.0, 1.0).astype(np.float32)


def render_sar(cover: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """``(2, h, w)`` float32 linear power: VV and VH with gamma speckle.

    Four looks, which is what a Sentinel-1 GRD product carries -- enough speckle
    to look like radar and little enough that the water threshold still works.
    """
    scale = np.zeros((SIZE, SIZE), dtype=np.float32)
    for class_id, (_reflectance, sar) in SIGNATURES.items():
        scale[cover == class_id] = sar

    vv = rng.gamma(4.0, scale / 4.0 * 0.06).astype(np.float32)
    vh = rng.gamma(4.0, scale / 4.0 * 0.02).astype(np.float32)  # cross-pol is weaker
    # A handful of corner reflectors: ships, pylons, metal roofs.
    for _ in range(12):
        y, x = rng.integers(0, SIZE, 2)
        vv[y : y + 2, x : x + 2] *= float(rng.uniform(20.0, 60.0))
    return np.stack([vv, vh]).astype(np.float32)


def apply_changes(cover: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The later date: a new development, a cleared forest patch, a widened lake."""
    after = cover.copy()
    _town(after, rng, y0=int(SIZE * 0.42), x0=int(SIZE * 0.70), rows=3, cols=4)
    after[int(SIZE * 0.16) : int(SIZE * 0.30), int(SIZE * 0.10) : int(SIZE * 0.26)] = BARE
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    lake = ((yy - SIZE * 0.78) / (SIZE * 0.125)) ** 2 + ((xx - SIZE * 0.17) / (SIZE * 0.16)) ** 2
    after[lake < 1.0] = WATER
    return after


def save_png(array: np.ndarray, path: Path, *, gamma: float = 0.85) -> Path:
    """Write the RGB bands as an 8-bit PNG: one common stretch, then gamma.

    Stretching each band independently is what a GIS does when you ask for
    "percentile per band", and it wrecks a synthetic scene: the blue band of a
    vegetated landscape has almost no variance, so normalising it to full range
    turns the whole picture neon. A *common* stretch keeps the band ratios --
    which is what makes vegetation read as olive rather than as green paint --
    and the gamma lifts the shadows the way a true-colour product does.
    """
    from PIL import Image

    rgb = np.asarray(array[:3], dtype=np.float32)
    low, high = np.percentile(rgb, (1.0, 99.2))
    stretched = np.clip((rgb - low) / max(1e-9, high - low), 0.0, 1.0) ** gamma
    hwc = np.transpose((stretched * 255).round().astype(np.uint8), (1, 2, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(hwc).save(path)
    return path


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = Path(__file__).resolve().parent.parent / "demo"
    out.mkdir(parents=True, exist_ok=True)

    before_cover = land_cover(rng)
    after_cover = apply_changes(before_cover, rng)

    # The *same* texture seed for both dates. Two acquisitions of one place share
    # their field boundaries, tree crowns and soil grain; rendering them from
    # independent noise would put a small difference in every pixel, and the
    # change map would come back speckled with regions that correspond to
    # nothing. Only the land cover and the sun angle differ.
    before = render(before_cover, np.random.default_rng(SEED + 1))
    # A different sun angle on the later date: the distractor a naive difference
    # reports as change across the whole scene.
    after = render(after_cover, np.random.default_rng(SEED + 1), brightness=1.09)

    save_png(before, out / "city_2019.png")
    save_png(after, out / "city_2021.png")

    # Four-band versions for the index demo: with a real NIR band the tools
    # compute NDVI and NDWI rather than the RGB proxy.
    np.save(out / "city_2019_4band.npy", before)
    np.save(out / "coast_optical_4band.npy", before)
    np.save(out / "coast_sar_vv_vh.npy", render_sar(before_cover, np.random.default_rng(SEED + 3)))

    print(f"demo scenes written to {out}")
    for path in sorted(out.iterdir()):
        print(f"  {path.name:28} {path.stat().st_size / 1024:8.0f} KB")


if __name__ == "__main__":
    main()
