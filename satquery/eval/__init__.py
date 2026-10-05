"""Evaluation harness and reporting."""

from .._lazy import lazy_getattr

_TABLE = {
    "evaluate": ".harness:evaluate",
    "save_run": ".harness:save_run",
    "load_run": ".harness:load_run",
    "run_matrix": ".harness:run_matrix",
    "render_markdown": ".report:render_markdown",
    "render_comparison": ".report:render_comparison",
}

__all__ = sorted(_TABLE)
__getattr__ = lazy_getattr(__name__, _TABLE)
