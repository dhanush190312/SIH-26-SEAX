"""The workspace: registered images, declared pairs, answered queries.

The session exists because of invariant 01. The router never sees pixels -- it
sees ``image_id`` strings and the validator's text summary, and the *executor*
resolves an id back to a file when it calls a tool. That indirection needs
somewhere to live, and this is it: an id -> file table with the validation report
attached, plus the pairs the user has declared and the queries already answered.

Everything is on disk under ``<runs_dir>/sessions/<session_id>/``:

    images/     the registered rasters, copied in under their id
    previews/   the display-transformed PNG each image renders to
    artifacts/  masks, overlays and index maps a query produced
    queries/    one JSON per answered query: result + trace
    session.json  the manifest

Copying uploads in rather than referencing them where they lie is deliberate:
the API accepts a byte stream with no path at all, a demo must survive the user
deleting their download folder, and the PDF report has to be able to embed the
exact image the answer was derived from an hour later.

A session is a plain directory, so "reset the demo" is ``rm -rf`` and "ship the
evidence" is a zip. There is no database and no server state beyond the
filesystem -- a restarted API picks up every session it left behind.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping

from ..config import SatQueryConfig, get_config
from ..contracts import ImageRef, SatQueryError
from ..log import get_logger
from .render import render_preview
from .trace import Trace, jsonable, utc_now
from .validate import ImageReport, PairReport, load_image, validate_image, validate_pair

__all__ = ["Session", "SessionStore", "SessionError", "DEFAULT_SESSION"]

_LOG = get_logger(__name__)

DEFAULT_SESSION = "default"

_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


class SessionError(SatQueryError):
    """A session, image or pair was asked for and does not exist."""


def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _safe(name: str) -> str:
    """Filesystem- and URL-safe form of a user-supplied name.

    Uploads carry attacker-controlled filenames; this is the only place one
    reaches the filesystem, so path separators and traversal are stripped here
    rather than trusted anywhere downstream.
    """
    cleaned = _SAFE_ID.sub("_", Path(str(name)).name).strip("._")
    return cleaned or "file"


def _companions(src: Path) -> list[Path]:
    """Sidecar files that must travel with a raster.

    Two shapes exist. ``scene.tif`` written without rasterio becomes
    ``scene.tif.npy`` + ``scene.tif.json``, and a caller may hold either path;
    whichever one is registered, both files have to land in the workspace or the
    georeferencing is lost on copy.
    """
    out: list[Path] = []
    for candidate in (Path(str(src) + ".npy"), Path(str(src) + ".json")):
        if candidate.exists():
            out.append(candidate)
    if src.suffix.lower() == ".npy":
        json_side = src.with_suffix(".json")
        if json_side.exists():
            out.append(json_side)
    return out


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """One registered image: where it lives, and what the validator made of it."""

    image_id: str
    path: str
    original_name: str
    report: ImageReport
    preview: str | None = None
    registered_at: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.report.ok

    @property
    def modality(self) -> str:
        return self.report.modality

    def as_ref(self) -> ImageRef:
        return self.report.as_ref()

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "path": self.path,
            "original_name": self.original_name,
            "preview": self.preview,
            "registered_at": self.registered_at,
            "validation": self.report.to_dict(),
            "meta": dict(self.meta),
        }


@dataclass(frozen=True, slots=True)
class PairRecord:
    """Two image ids the user declared as a pair, plus the geometric verdict."""

    pair_id: str
    image_a: str
    image_b: str
    report: PairReport
    registered_at: str = ""

    @property
    def ok(self) -> bool:
        return self.report.ok

    @property
    def pair_type(self) -> str:
        return self.report.pair_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "image_a": self.image_a,
            "image_b": self.image_b,
            "registered_at": self.registered_at,
            "check": self.report.to_dict(),
        }


class Session:
    """One workspace directory, held open.

    Not thread-safe by design: the manifest is rewritten whole on every mutation,
    which is correct for one user driving one GUI and would need a lock for
    concurrent writers. That trade is worth stating rather than discovering --
    the alternative is a database in a prototype that has to run offline.
    """

    def __init__(self, session_id: str, root: Path, config: SatQueryConfig | None = None) -> None:
        self.session_id = session_id
        self.root = Path(root)
        self.config = config or get_config()
        self.images_dir = self.root / "images"
        self.previews_dir = self.root / "previews"
        self.artifacts_dir = self.root / "artifacts"
        self.queries_dir = self.root / "queries"
        for d in (self.images_dir, self.previews_dir, self.artifacts_dir, self.queries_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._images: dict[str, ImageRecord] = {}
        self._pairs: dict[str, PairRecord] = {}
        self._queries: list[str] = []
        self._load_manifest()

    # -- images -------------------------------------------------------------- #

    def add_image(
        self,
        source: str | Path | BinaryIO | bytes,
        *,
        filename: str | None = None,
        modality: str | None = None,
        image_id: str | None = None,
        copy: bool = True,
    ) -> ImageRecord:
        """Register a raster and validate it. Returns the record either way.

        ``source`` is a path, raw bytes or an open binary stream (the API hands
        over an upload's stream). A file that fails validation is still
        registered: the GUI has to be able to show the user *why* their upload
        was refused, next to its name.
        """
        image_id = image_id or _short_id("img")
        from_path = isinstance(source, (str, Path))
        # The *stored* name always keeps the source file's own suffix. Taking it
        # from a caller-supplied ``filename`` instead would write a .npy sidecar
        # under a .tif name, and the reader would then hand it to PIL and fail.
        # ``filename`` survives as ``original_name``, which is what the GUI shows.
        name = _safe(Path(str(source)).name if from_path else (filename or "upload.tif"))
        target = self.images_dir / f"{image_id}_{name}"

        if from_path:
            src = Path(source)
            if not (src.exists() or Path(str(src) + ".npy").exists()):
                raise SessionError(f"no such image: {src}")
            if copy:
                if src.exists():
                    shutil.copyfile(src, target)
                for companion in _companions(src):
                    shutil.copyfile(
                        companion, Path(str(target) + companion.name[len(src.name):])
                        if companion.name.startswith(src.name)
                        else target.with_suffix(companion.suffix)
                    )
            else:
                target = src
        else:
            payload = source if isinstance(source, bytes) else source.read()
            target.write_bytes(payload)

        report = validate_image(target, image_id=image_id, declared_modality=modality)
        preview = None
        if report.ok:
            try:
                preview = str(
                    render_preview(
                        load_image(target),
                        self.previews_dir / f"{image_id}.png",
                        modality=report.modality,
                        max_side=self.config.max_side,
                    )
                )
            except Exception as exc:  # a preview is a convenience, never a blocker
                _LOG.warning("preview for %s failed: %s", image_id, exc)

        record = ImageRecord(
            image_id=image_id,
            path=str(target),
            original_name=str(filename or (Path(str(source)).name if from_path else name)),
            report=report,
            preview=preview,
            registered_at=utc_now(),
        )
        self._images[image_id] = record
        self._save_manifest()
        return record

    def image(self, image_id: str) -> ImageRecord:
        try:
            return self._images[image_id]
        except KeyError:
            raise SessionError(
                f"unknown image {image_id!r}; registered: {', '.join(self._images) or 'none'}"
            ) from None

    @property
    def images(self) -> tuple[ImageRecord, ...]:
        return tuple(self._images.values())

    # -- pairs --------------------------------------------------------------- #

    def add_pair(self, image_a: str, image_b: str, *, pair_id: str | None = None) -> PairRecord:
        """Declare two registered images as a pair and run the geometric check.

        ``image_a`` is the earlier date for bi-temporal work and the optical
        member for cross-modal work. Both conventions are enforced downstream by
        prompt wording, so getting the order wrong here produces a backwards
        change description -- the GUI labels the slots accordingly.
        """
        a, b = self.image(image_a), self.image(image_b)
        pair_id = pair_id or _short_id("pair")
        report = validate_pair(a.report, b.report, pair_id=pair_id)
        record = PairRecord(
            pair_id=pair_id,
            image_a=a.image_id,
            image_b=b.image_id,
            report=report,
            registered_at=utc_now(),
        )
        self._pairs[pair_id] = record
        self._save_manifest()
        return record

    def pair(self, pair_id: str) -> PairRecord:
        try:
            return self._pairs[pair_id]
        except KeyError:
            raise SessionError(
                f"unknown pair {pair_id!r}; registered: {', '.join(self._pairs) or 'none'}"
            ) from None

    @property
    def pairs(self) -> tuple[PairRecord, ...]:
        return tuple(self._pairs.values())

    def pair_for(self, image_ids: Iterable[str]) -> PairRecord | None:
        """An existing pair over exactly these two ids, in either order."""
        wanted = set(image_ids)
        if len(wanted) != 2:
            return None
        for record in self._pairs.values():
            if {record.image_a, record.image_b} == wanted:
                return record
        return None

    # -- artefacts and queries ------------------------------------------------ #

    def artifact_path(self, query_id: str, name: str) -> Path:
        """Where a tool writes a mask, overlay or index map for one query."""
        d = self.artifacts_dir / _safe(query_id)
        d.mkdir(parents=True, exist_ok=True)
        return d / _safe(name)

    def save_query(self, result: Any, trace: Trace | None = None) -> Path:
        """Persist one answered query. ``result`` is anything with ``to_dict``."""
        query_id = getattr(result, "query_id", None) or _short_id("q")
        payload: dict[str, Any] = {
            "result": jsonable(result),
            "trace": trace.to_dict() if trace is not None else None,
            "saved_at": utc_now(),
        }
        path = self.queries_dir / f"{query_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if query_id not in self._queries:
            self._queries.append(query_id)
            self._save_manifest()
        return path

    def load_query(self, query_id: str) -> dict[str, Any]:
        path = self.queries_dir / f"{_safe(query_id)}.json"
        if not path.exists():
            raise SessionError(f"unknown query {query_id!r} in session {self.session_id!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def query_ids(self) -> tuple[str, ...]:
        return tuple(self._queries)

    def new_query_id(self) -> str:
        return _short_id("q")

    # -- manifest ------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "root": str(self.root),
            "images": [r.to_dict() for r in self._images.values()],
            "pairs": [r.to_dict() for r in self._pairs.values()],
            "queries": list(self._queries),
        }

    def _manifest_path(self) -> Path:
        return self.root / "session.json"

    def _save_manifest(self) -> None:
        self._manifest_path().write_text(
            json.dumps(jsonable(self.to_dict()), indent=2), encoding="utf-8"
        )

    def _load_manifest(self) -> None:
        """Rebuild from disk, re-validating rather than trusting the stored report.

        Re-running validation on load costs one raster read per image and buys
        the guarantee that a manifest edited by hand, or written by an older
        version of the code, cannot inject a stale verdict into a fresh answer.
        """
        path = self._manifest_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _LOG.warning("ignoring corrupt session manifest %s: %s", path, exc)
            return
        for raw in data.get("images", ()):
            image_path = Path(raw.get("path", ""))
            if not (image_path.exists() or Path(str(image_path) + ".npy").exists()):
                _LOG.warning("dropping %s: file is gone", raw.get("image_id"))
                continue
            declared = raw.get("validation", {}).get("modality_source") == "declared"
            report = validate_image(
                image_path,
                image_id=str(raw["image_id"]),
                declared_modality=raw.get("validation", {}).get("modality") if declared else None,
            )
            self._images[str(raw["image_id"])] = ImageRecord(
                image_id=str(raw["image_id"]),
                path=str(image_path),
                original_name=str(raw.get("original_name", image_path.name)),
                report=report,
                preview=raw.get("preview"),
                registered_at=str(raw.get("registered_at", "")),
            )
        for raw in data.get("pairs", ()):
            a, b = str(raw.get("image_a", "")), str(raw.get("image_b", ""))
            if a not in self._images or b not in self._images:
                continue
            pair_id = str(raw["pair_id"])
            self._pairs[pair_id] = PairRecord(
                pair_id=pair_id,
                image_a=a,
                image_b=b,
                report=validate_pair(self._images[a].report, self._images[b].report, pair_id=pair_id),
                registered_at=str(raw.get("registered_at", "")),
            )
        self._queries = [str(q) for q in data.get("queries", ())]


class SessionStore:
    """Opens sessions by id under one root, creating them on first use.

    The GUI and the CLI share this, so ``satquery query`` can answer against
    images the web app registered a minute earlier -- one workspace, two front
    ends, no import step.
    """

    def __init__(self, root: str | Path | None = None, config: SatQueryConfig | None = None) -> None:
        self.config = config or get_config()
        self.root = Path(root or os.environ.get("SATQUERY_SESSIONS_DIR") or self.config.runs_dir / "sessions")
        self.root.mkdir(parents=True, exist_ok=True)
        self._open: dict[str, Session] = {}

    def get(self, session_id: str = DEFAULT_SESSION) -> Session:
        sid = _safe(session_id)
        if sid not in self._open:
            self._open[sid] = Session(sid, self.root / sid, config=self.config)
        return self._open[sid]

    def create(self, session_id: str | None = None) -> Session:
        return self.get(session_id or _short_id("s"))

    def list_ids(self) -> tuple[str, ...]:
        return tuple(sorted(p.name for p in self.root.iterdir() if p.is_dir()))

    def delete(self, session_id: str) -> bool:
        sid = _safe(session_id)
        path = self.root / sid
        self._open.pop(sid, None)
        if not path.exists():
            return False
        shutil.rmtree(path)
        return True
