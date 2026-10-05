"""Paths, seeds and device policy.

One rule drives the layout: nothing in the repo may hardcode a dataset path.
Benchmarks live on the laptop under ``D:\\SatQueryAI\\data`` during development
and under ``/kaggle/input`` or ``/content/data`` on the GPU boxes, so every
adapter asks :meth:`SatQueryConfig.dataset_root` instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "SatQueryConfig",
    "get_config",
    "set_config",
    "repo_root",
    "load_env_file",
    "ENV_FILE",
]

#: Name of the optional key/value file loaded on import. Gitignored, and it is
#: the only place a credential should ever live in this project.
ENV_FILE = ".env"

#: T4 (SM75) cannot do bf16 and has no FlashAttention-2. Anything that builds a
#: torch dtype must consult this rather than defaulting to bf16.
T4_SAFE_DTYPE = "float16"
T4_SAFE_ATTENTION = "sdpa"


def repo_root() -> Path:
    """The repository root (parent of the ``satquery`` package)."""
    return Path(__file__).resolve().parent.parent


def load_env_file(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Read ``KEY=value`` lines from a ``.env`` file into the process environment.

    A twelve-line parser rather than ``python-dotenv``, because the package rule
    is that numpy is the only hard dependency and a credential loader is not
    worth breaking it for. It understands what a ``.env`` actually contains:
    comments, blank lines, ``export`` prefixes, and values in single or double
    quotes. It does not do variable interpolation -- an API key containing ``$``
    must survive verbatim.

    **A real environment variable always wins.** ``override`` exists for tests
    and is off by default, so ``MISTRAL_API_KEY=... satquery serve`` beats a
    stale value in the file rather than being silently replaced by it.

    Returns the keys it actually set, so a caller can log *which* names were
    loaded. It never returns or logs a value: this function exists to move
    credentials, and the one thing it must not do is print one.
    """
    candidates = (
        [Path(path)]
        if path is not None
        else [Path.cwd() / ENV_FILE, repo_root() / ENV_FILE]
    )
    loaded: dict[str, str] = {}
    for candidate in candidates:
        try:
            if not candidate.is_file():
                continue
            text = candidate.read_text(encoding="utf-8")
        except OSError:
            continue  # unreadable .env is not a reason to fail an import
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if not key or (key in os.environ and not override):
                continue
            os.environ[key] = value
            loaded[key] = candidate.name
    return loaded


# Loaded once, at import, because every entry point (CLI, API, notebook, tests)
# reaches this module and none of them should have to remember to call it.
load_env_file()


@dataclass(frozen=True, slots=True)
class SatQueryConfig:
    """Resolved environment for a run.

    Environment variables (all optional):

    ``SATQUERY_DATA_ROOT``   where benchmark downloads live (default ``<repo>/data``)
    ``SATQUERY_RUNS_DIR``    where run JSON + reports are written (default ``<repo>/runs``)
    ``SATQUERY_CACHE_DIR``   scratch for decoded tiles / manifests (default ``<data_root>/.cache``)
    ``SATQUERY_SEED``        global seed (default 1337)
    ``SATQUERY_DEVICE``      ``auto`` | ``cpu`` | ``cuda`` (default ``auto``)
    ``SATQUERY_MAX_SIDE``    longest image side handed to a VLM (default 768)
    """

    data_root: Path
    runs_dir: Path
    cache_dir: Path
    seed: int = 1337
    device: str = "auto"
    max_side: int = 768
    dataset_paths: Mapping[str, str] = field(default_factory=dict)

    # -- construction -------------------------------------------------------- #

    @classmethod
    def from_env(cls, **overrides: Any) -> "SatQueryConfig":
        root = repo_root()
        data_root = Path(os.environ.get("SATQUERY_DATA_ROOT") or root / "data")
        runs_dir = Path(os.environ.get("SATQUERY_RUNS_DIR") or root / "runs")
        cache_dir = Path(os.environ.get("SATQUERY_CACHE_DIR") or data_root / ".cache")
        cfg = cls(
            data_root=data_root,
            runs_dir=runs_dir,
            cache_dir=cache_dir,
            seed=int(os.environ.get("SATQUERY_SEED", "1337")),
            device=os.environ.get("SATQUERY_DEVICE", "auto"),
            max_side=int(os.environ.get("SATQUERY_MAX_SIDE", "768")),
        )
        if overrides:
            from dataclasses import replace as _replace

            cfg = _replace(cfg, **overrides)
        return cfg

    # -- paths --------------------------------------------------------------- #

    def dataset_root(self, name: str) -> Path:
        """Root directory for one dataset.

        ``SATQUERY_DATA_<NAME>`` overrides an individual dataset, which is how a
        Kaggle notebook points ``rsvqa_lr`` at ``/kaggle/input/rsvqa-lr`` without
        moving anything else.
        """
        env_key = f"SATQUERY_DATA_{name.upper()}"
        explicit = os.environ.get(env_key) or self.dataset_paths.get(name)
        return Path(explicit) if explicit else self.data_root / name

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def ensure_dirs(self) -> "SatQueryConfig":
        for p in (self.data_root, self.runs_dir, self.cache_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_root": str(self.data_root),
            "runs_dir": str(self.runs_dir),
            "cache_dir": str(self.cache_dir),
            "seed": self.seed,
            "device": self.device,
            "max_side": self.max_side,
        }

    # -- device -------------------------------------------------------------- #

    def resolve_device(self) -> str:
        """``cuda`` when torch sees a GPU and the config allows it, else ``cpu``."""
        if self.device != "auto":
            return self.device
        try:  # torch is an optional extra
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


_ACTIVE: SatQueryConfig | None = None


def get_config() -> SatQueryConfig:
    """Process-wide config, built from the environment on first use."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = SatQueryConfig.from_env()
    return _ACTIVE


def set_config(cfg: SatQueryConfig) -> SatQueryConfig:
    """Install a config (tests use this to point at a tmp data root)."""
    global _ACTIVE
    _ACTIVE = cfg
    return cfg
