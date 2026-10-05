"""Logging. One handler, one format, level from ``SATQUERY_LOG_LEVEL``."""

from __future__ import annotations

import logging
import os
import sys

__all__ = ["get_logger"]

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("SATQUERY_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    root = logging.getLogger("satquery")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level, logging.INFO))
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Logger under the ``satquery`` namespace (``name`` may be ``__name__``)."""
    _configure()
    if name.startswith("satquery."):
        name = name[len("satquery.") :]
    return logging.getLogger(f"satquery.{name}" if name != "satquery" else "satquery")
