"""Lazy submodule attribute access (PEP 562).

Package ``__init__`` files declare ``{public_name: "module:attr"}`` and get a
module-level ``__getattr__`` that imports on first touch. Two reasons this is
worth the indirection:

* ``import satquery.io`` must not drag in rasterio, and ``satquery.metrics``
  must not import every metric to give you one -- the CLI lists datasets on a
  laptop with no extras installed.
* A broken or not-yet-written module cannot take down its siblings, which is
  what makes the package usable while it is still being filled in.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping

__all__ = ["lazy_getattr"]


def lazy_getattr(package: str, table: Mapping[str, str]) -> Callable[[str], Any]:
    """Build a ``__getattr__`` for ``package`` from ``{name: 'module:attr'}``.

    ``module`` may be relative (``".vqa"``) or absolute.
    """

    def __getattr__(name: str) -> Any:
        try:
            target = table[name]
        except KeyError:
            raise AttributeError(f"module {package!r} has no attribute {name!r}") from None
        module_name, _, attr = target.partition(":")
        module = importlib.import_module(module_name, package)
        value = getattr(module, attr) if attr else module
        globals_ = importlib.import_module(package).__dict__
        globals_[name] = value  # cache, so the import cost is paid once
        return value

    return __getattr__
