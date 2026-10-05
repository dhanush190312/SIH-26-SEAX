"""HTTP surface: ``/api/v1`` for users, ``/tools/v1`` for the tool contract, ``/`` for the GUI."""

from .._lazy import lazy_getattr

_TABLE = {
    "create_app": ".app:create_app",
    "run": ".app:run",
}

__all__ = sorted(_TABLE)
__getattr__ = lazy_getattr(__name__, _TABLE)
