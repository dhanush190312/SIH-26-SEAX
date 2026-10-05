"""Shared test fixtures.

Everything here is synthetic. No test may download a dataset or model: the whole
suite has to run on a laptop with numpy + pytest in a couple of seconds, because
that is the only way it gets run before every Colab session.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from satquery.config import SatQueryConfig, get_config, set_config
from satquery.contracts import Box, ImageRef, Sample
from satquery.models.dummy import ConstantPredictor, DummyPredictor, OraclePredictor

SEED = 1337


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture
def tmp_config(tmp_path: Path):
    """Install a config rooted in ``tmp_path`` for the duration of a test."""
    previous = get_config()
    cfg = SatQueryConfig(
        data_root=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        seed=SEED,
        device="cpu",
        max_side=768,
    ).ensure_dirs()
    set_config(cfg)
    try:
        yield cfg
    finally:
        set_config(previous)


# --------------------------------------------------------------------------- #
# Synthetic imagery
# --------------------------------------------------------------------------- #


@pytest.fixture
def optical_array() -> Callable[..., np.ndarray]:
    """Factory for a plausible optical stack: ``(bands, h, w)`` float32 reflectance.

    Band 0 is a smooth gradient, the rest are structured noise, and a bright
    square is stamped in so stretches and composites have something to bite on.
    """

    def _make(bands: int = 4, height: int = 64, width: int = 64, seed: int = SEED) -> np.ndarray:
        g = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        base = (xx / max(width - 1, 1)) * 0.4 + (yy / max(height - 1, 1)) * 0.2
        stack = np.stack(
            [base * (1.0 + 0.1 * b) + g.normal(0, 0.02, size=(height, width)) for b in range(bands)]
        ).astype(np.float32)
        stack[:, height // 4 : height // 2, width // 4 : width // 2] += 0.35
        return np.clip(stack, 0.0, 1.0).astype(np.float32)

    return _make


@pytest.fixture
def sar_array() -> Callable[..., np.ndarray]:
    """Factory for SAR *linear power* (not dB): gamma-distributed speckle.

    Includes a low-backscatter patch standing in for water, so dB conversion,
    speckle filtering and water masking all have a signal to find.
    """

    def _make(bands: int = 2, height: int = 64, width: int = 64, seed: int = SEED) -> np.ndarray:
        g = np.random.default_rng(seed)
        out = g.gamma(shape=4.0, scale=0.05, size=(bands, height, width)).astype(np.float32)
        out[:, : height // 3, : width // 3] *= 0.02  # water: ~-17 dB darker
        return out

    return _make


@pytest.fixture
def write_png() -> Callable[..., Path]:
    """Write a ``(band,h,w)`` float array in [0,1] (or uint8) as a PNG."""

    def _write(path: str | Path, array: np.ndarray) -> Path:
        from PIL import Image  # optional extra, present on the dev laptop

        arr = np.asarray(array)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)
        hwc = np.transpose(arr, (1, 2, 0))
        if hwc.shape[2] == 1:
            img = Image.fromarray(hwc[:, :, 0], mode="L")
        else:
            img = Image.fromarray(hwc[:, :, :3], mode="RGB")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
        return p

    return _write


@pytest.fixture
def write_raster() -> Callable[..., Path]:
    """Write a georeferenced raster via ``satquery.io.raster.write_geotiff``.

    Uses rasterio when installed and the documented ``.npy`` + JSON sidecar
    fallback when not, so the geo code paths are still exercised on the laptop.
    """

    def _write(path: str | Path, array: np.ndarray, **kwargs: Any) -> Path:
        from satquery.io.raster import write_geotiff

        return write_geotiff(path, np.asarray(array), **kwargs)

    return _write


@pytest.fixture
def rgb_scene() -> Callable[..., np.ndarray]:
    """A ``(3, h, w)`` uint8 scene with named regions the tools should find.

    Deliberately blocky rather than realistic: every test that asserts "water is
    12.25% of this scene" needs the truth to be a number computable by hand, and
    a synthetic block is the only input for which that is true. ``shift`` moves
    the built-up block, which is the change a bi-temporal test measures.
    """

    def _make(size: int = 128, shift: int = 0, noise: float = 0.0, seed: int = SEED) -> np.ndarray:
        g = np.random.default_rng(seed)
        img = np.zeros((3, size, size), dtype=np.float32)
        img[0], img[1], img[2] = 110.0, 130.0, 90.0  # vegetated background
        quarter = size // 4
        img[:, :quarter, :quarter] = np.array([40.0, 70.0, 140.0])[:, None, None]  # water
        y0, x0 = size // 2, size // 2 + shift
        img[:, y0 : y0 + quarter, x0 : min(size, x0 + quarter)] = 150.0  # built-up
        if noise:
            img += g.normal(0.0, noise, img.shape).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)

    return _make


@pytest.fixture
def scene_file(tmp_path, write_png, rgb_scene) -> Callable[..., Path]:
    """``rgb_scene`` written to a PNG and returned as a path."""

    def _write(name: str = "scene.png", **kwargs: Any) -> Path:
        return write_png(tmp_path / name, rgb_scene(**kwargs))

    return _write


@pytest.fixture
def session(tmp_config):
    """An empty session rooted in the test's temporary runs directory."""
    from satquery.agent.session import SessionStore

    return SessionStore(config=tmp_config).get("test")


# --------------------------------------------------------------------------- #
# Samples and predictors
# --------------------------------------------------------------------------- #


@pytest.fixture
def make_image_ref() -> Callable[..., ImageRef]:
    def _make(image_id: str = "img1", path: str = "img1.png", **kw: Any) -> ImageRef:
        return ImageRef(image_id=image_id, path=path, **kw)

    return _make


@pytest.fixture
def make_sample(make_image_ref) -> Callable[..., Sample]:
    """Factory for a valid :class:`Sample` of any task, with sane defaults."""

    def _make(sample_id: str = "s1", task: str = "vqa", **kw: Any) -> Sample:
        n_images = 2 if task in ("change_vqa", "change_caption", "change_map", "fusion_multilabel") else 1
        images = kw.pop(
            "images",
            tuple(make_image_ref(f"{sample_id}_i{i}", f"{sample_id}_i{i}.png") for i in range(n_images)),
        )
        defaults: dict[str, Any] = {"dataset": "synthetic", "split": "test"}
        if task in ("vqa", "change_vqa"):
            defaults.update(question="how many buildings?", answer="2")
        elif task in ("caption", "change_caption"):
            defaults.update(references=("a residential area with several buildings",))
        elif task == "ground":
            defaults.update(question="the large building", boxes=(Box(10, 10, 40, 40, "building"),))
        elif task == "fusion_multilabel":
            defaults.update(labels=("urban_fabric", "water_bodies"))
        defaults.update(kw)
        return Sample(sample_id=sample_id, task=task, images=images, **defaults)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def oracle() -> OraclePredictor:
    return OraclePredictor()


@pytest.fixture
def dummy() -> DummyPredictor:
    return DummyPredictor(seed=SEED, accuracy=0.5)


@pytest.fixture
def constant() -> ConstantPredictor:
    return ConstantPredictor(text="yes")


# --------------------------------------------------------------------------- #
# Assertions
# --------------------------------------------------------------------------- #


@pytest.fixture
def capture_logs() -> Callable[..., list[logging.LogRecord]]:
    """Records emitted by one ``satquery.*`` logger, as a list that fills up.

    ``satquery.log`` sets ``propagate=False`` on the package logger so library
    output never doubles up under an application's root handler. The cost is
    that pytest's ``caplog``, which listens on the root logger, sees nothing at
    all -- a warning test written with ``caplog`` passes vacuously or fails
    confusingly. Attaching a sink directly to the named logger is the least
    invasive fix and leaves the frozen logging setup alone.
    """
    attached: list[tuple[logging.Logger, logging.Handler]] = []

    def _capture(name: str, level: int = logging.WARNING) -> list[logging.LogRecord]:
        records: list[logging.LogRecord] = []

        class _Sink(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        logger = logging.getLogger(name)
        handler = _Sink(level=level)
        logger.addHandler(handler)
        attached.append((logger, handler))
        return records

    yield _capture
    for logger, handler in attached:
        logger.removeHandler(handler)


@pytest.fixture
def assert_metrics_close() -> Callable[..., None]:
    def _assert(got: dict[str, float], expected: dict[str, float], tol: float = 1e-6) -> None:
        for key, want in expected.items():
            assert key in got, f"missing metric {key!r}; got {sorted(got)}"
            assert abs(got[key] - want) <= tol, f"{key}: got {got[key]!r}, want {want!r}"

    return _assert


def write_json(path: str | Path, payload: Any) -> Path:
    """Helper for dataset-fixture builders (not a fixture: importable from tests)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return p
