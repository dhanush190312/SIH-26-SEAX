"""RSVQA-LR and RSVQA-HR adapters (Lobry et al., TGRS 2020).

RSVQA is the headline single-image VQA benchmark for this challenge, so the
parser's assumptions are spelled out here rather than left implicit.

The Zenodo distribution ships, *per split*, a triple of JSON files plus one flat
folder of numbered image tiles::

    RSVQA-LR (zenodo 6344334)          RSVQA-HR (zenodo 6344367)
    LR_split_test_questions.json       USGS_split_test_questions.json
    LR_split_test_answers.json         USGS_split_test_answers.json
    LR_split_test_images.json          USGS_split_test_images.json
    Images_LR/<img_id>.tif             Images/Data/<img_id>.tif

The three tables are relational, not row-aligned -- that is the part that is
easy to get wrong:

* ``questions[i]`` carries ``id``, ``question``, ``type`` and ``img_id``.
* ``answers[i]`` carries ``question_id``, ``answer`` and ``active``. The join is
  answer -> question, so answers are indexed by ``question_id`` here. (A
  ``answers_ids`` list on the question row also appears in the wild; it is
  honoured as a fallback.)
* ``images[i]`` carries ``questions_ids`` and ``active``, i.e. the image table is
  where whole tiles get withdrawn.

``active`` is the flag that decides what is scorable. The reference loaders --
`syvlo/RSVQA <https://github.com/syvlo/RSVQA>`_ and ConfigILM's
``RSVQALR_DataSet`` -- keep only questions belonging to an *active* image and
only answers whose own ``active`` is true. Inactive rows are pairs the question
generator produced but the authors withdrew, so scoring them means comparing
against ground truth no published number used. We drop them and *count* the
drops (:attr:`load_stats`, surfaced through :meth:`describe`) rather than
swallowing them silently, because "why is my test split smaller than the paper's"
is otherwise unanswerable.

Ids are used as keys, never as list positions. The reference loader can index
positionally because ``id == index`` holds on the official download; a
re-exported or filtered split breaks that, and a silent off-by-one in the
question/answer join would look like a bad model rather than a bad parse.

RSVQA-HR ships two test sets: ``test`` (tiles from regions adjacent to training)
and ``test_phili`` (Philadelphia, a city absent from training). The second is the
honest generalisation number, so it is exposed as split ``test2`` per the Phase 1
spec and mapped back to the ``test_phili`` filename token here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..contracts import Sample
from ..log import get_logger
from .base import BaseDataset, first_existing, read_json

__all__ = ["RSVQALRDataset", "RSVQAHRDataset"]

log = get_logger(__name__)

#: Sub-directories probed for the annotation triple. The Zenodo files land beside
#: the image folder; the RSVQA reference training scripts and most Kaggle mirrors
#: move them into ``text/`` or ``jsons/`` (ConfigILM documents the latter).
_JSON_DIRS: tuple[str, ...] = ("", "text", "jsons", "json", "annotations", "Questions")

#: Probed once per load to name the image files. RSVQA ships uncompressed
#: GeoTIFF; benchmark-mode re-exports (and ConfigILM's fixtures) use PNG.
_IMAGE_EXTENSIONS: tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# VERIFY: the raw ``type`` strings are recalled from the RSVQA question
# generator, not read off a download; only the four *categories* per collection
# are confirmed (from the paper). Spelling variants are folded here rather than
# in the metric, so a mismatch costs a per-type row in the report and never a
# wrong overall accuracy.
_TYPE_ALIASES: Mapping[str, str] = {
    "rural urban": "rural_urban",
    "urban rural": "rural_urban",
    "ruralurban": "rural_urban",
    "comp": "comparison",
    "compare": "comparison",
    "counting": "count",
    "presence absence": "presence",
}


def _normalize_question_type(raw: Any) -> str:
    """Fold the annotation's ``type`` onto the vocabulary the metric slices on.

    Returns ``"unknown"`` for a missing type rather than ``"all"``: ``"all"`` is
    :attr:`Sample.question_type`'s own name for "this dataset has no types", and
    conflating the two would hide a parse failure behind a plausible headline.
    """
    text = str(raw or "").strip().lower().replace("-", "_").replace("/", " ")
    text = " ".join(text.split())
    if text in _TYPE_ALIASES:
        return _TYPE_ALIASES[text]
    text = text.replace(" ", "_")
    return _TYPE_ALIASES.get(text, text) or "unknown"


def _entries(payload: Any, key: str) -> list[Any]:
    """Rows of one RSVQA table, tolerating the shapes seen in the wild.

    ``{"questions": [...]}`` is the official layout. A bare list, and a mapping
    of id -> row (with or without the outer ``questions`` key), both turn up in
    re-exports. An unrecognised shape returns ``[]``, which surfaces as
    BaseDataset's "parsed 0 samples" warning rather than a traceback.
    """
    if isinstance(payload, Mapping):
        for candidate in (key, key.rstrip("s")):
            if candidate in payload:
                payload = payload[candidate]
                break
    if isinstance(payload, Mapping):
        # id-keyed mapping; the row's own ``id`` field is authoritative, so the
        # mapping key is discarded rather than parsed.
        return list(payload.values())
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return list(payload)
    return []


def _row_id(row: Mapping[str, Any], position: int, *fields: str) -> int:
    """The row's integer id, falling back to its position in the table."""
    for field in fields:
        if field in row:
            try:
                return int(row[field])
            except (TypeError, ValueError):
                continue
    return position


def _is_active(row: Mapping[str, Any]) -> bool:
    """``active`` defaults to True: a re-export that dropped the column is usable."""
    return bool(row.get("active", True))


def _int_set(value: Any) -> set[int]:
    """Coerce ``questions_ids`` (a list, or a bare id in re-exports) to ints."""
    if value is None:
        return set()
    if isinstance(value, (int, str)):
        value = [value]
    out: set[int] = set()
    for item in value:
        try:
            out.add(int(item))
        except (TypeError, ValueError):
            continue
    return out


class _RSVQADataset(BaseDataset):
    """Shared parser for the two RSVQA collections.

    The subclasses differ only in file prefix, image folder, tile geometry and
    the question-type vocabulary; the relational join lives here exactly once so
    LR and HR cannot drift apart.
    """

    task = "vqa"

    #: filename stem, e.g. ``"LR_split"`` -> ``LR_split_test_questions.json``
    file_prefix: str = ""
    #: adapter split -> filename token (they differ only for HR's ``test2``)
    split_tokens: Mapping[str, str] = {}
    #: candidate image folders relative to :attr:`root`, most specific first
    image_dirs: tuple[str, ...] = ()
    #: the vocabulary the challenge scores per-type accuracy over
    question_types: tuple[str, ...] = ()
    #: what the tiles are; recorded on every ImageRef for the execution trace
    sensor: str = ""
    #: nominal tile edge in pixels, for the trace only -- never used to resize
    tile_size: int = 0

    def __init__(self, split: str | None = None, **kwargs: Any) -> None:
        super().__init__(split, **kwargs)
        self.split_token = self.split_tokens.get(self.split, self.split)
        #: Only the questions/answers pair is load-bearing, so only that pair is
        #: required; ``_images.json`` carries the tile-level ``active`` flag and
        #: several mirrors omit it (we then trust the answers' own flag).
        self.required_paths = tuple(
            self._json_name(kind) for kind in ("questions", "answers")
        )
        #: Parse bookkeeping, filled by :meth:`_load`. Empty until iterated.
        self.load_stats: dict[str, Any] = {}

    # -- file location ------------------------------------------------------- #

    def _json_name(self, kind: str) -> str:
        return f"{self.file_prefix}_{self.split_token}_{kind}.json"

    def _json_path(self, kind: str) -> Path | None:
        """Locate one annotation file across the known layout variants."""
        base = self._json_name(kind)
        candidates: list[Path] = []
        for directory in _JSON_DIRS:
            for name in (base, base.lower()):
                candidates.append(self.path(directory, name) if directory else self.path(name))
        return first_existing(*candidates)

    def missing_paths(self) -> list[str]:
        """Canonical filenames that are absent, not the variant paths probed.

        The error message has to name a file the user can go and download, so it
        reports ``LR_split_test_questions.json`` rather than the last of a dozen
        candidate paths that did not exist.
        """
        return [kind for kind in self.required_paths if self._json_path(_kind_of(kind)) is None]

    # -- image paths --------------------------------------------------------- #

    def _image_dir(self) -> Path:
        found = first_existing(*(self.path(d) for d in self.image_dirs))
        return found if found is not None else self._root

    def _image_extension(self, image_dir: Path, probe_id: int) -> str:
        """Resolve the tile extension once, by probing a single known id.

        Per-sample ``exists()`` is not affordable -- RSVQA-HR train is ~650k
        questions -- and the folder is homogeneous, so one probe stands in for
        all of them. Falls back to ``.tif`` (what Zenodo ships) so the path stays
        informative when only the annotations were downloaded.
        """
        for ext in _IMAGE_EXTENSIONS:
            if (image_dir / f"{probe_id}{ext}").exists():
                return ext
        return _IMAGE_EXTENSIONS[0]

    # -- loading ------------------------------------------------------------- #

    def _load(self) -> Iterable[Sample]:
        questions_path = self._json_path("questions")
        answers_path = self._json_path("answers")
        images_path = self._json_path("images")
        if questions_path is None or answers_path is None:  # pragma: no cover
            raise AssertionError("require_available() should have caught this")

        questions = _entries(read_json(questions_path), "questions")
        answers_by_question, answers_by_id = _index_answers(
            _entries(read_json(answers_path), "answers")
        )
        allowed_qids, inactive_images = _image_gate(
            _entries(read_json(images_path), "images") if images_path is not None else []
        )

        image_dir = self._image_dir()
        image_ext: str | None = None
        stats: dict[str, Any] = {
            "n_questions": len(questions),
            "dropped_inactive": 0,
            "dropped_inactive_question": 0,
            "dropped_inactive_image": 0,
            "dropped_inactive_answer": 0,
            "dropped_no_answer": 0,
            "dropped_malformed": 0,
            "n_images": 0,
            "unexpected_types": [],
        }
        seen_images: set[int] = set()
        unexpected: set[str] = set()

        for position, question in enumerate(questions):
            if not isinstance(question, Mapping):
                stats["dropped_malformed"] += 1
                continue
            if not _is_active(question):
                stats["dropped_inactive_question"] += 1
                stats["dropped_inactive"] += 1
                continue

            question_id = _row_id(question, position, "id", "question_id")
            try:
                image_id = int(question["img_id"])
            except (KeyError, TypeError, ValueError):
                stats["dropped_malformed"] += 1
                continue

            # Two independent ways the image table withdraws a tile: an explicit
            # inactive row, or simply not listing the question among an active
            # image's ``questions_ids`` (what the reference loader keys on).
            if image_id in inactive_images or (
                allowed_qids is not None and question_id not in allowed_qids
            ):
                stats["dropped_inactive_image"] += 1
                stats["dropped_inactive"] += 1
                continue

            question_text = str(question.get("question", "")).strip()
            if not question_text:  # Sample would reject a question-less vqa row
                stats["dropped_malformed"] += 1
                continue

            texts, reason = _answer_texts(question, question_id, answers_by_question, answers_by_id)
            if not texts:
                stats[reason] += 1
                if reason == "dropped_inactive_answer":
                    stats["dropped_inactive"] += 1
                continue

            qtype = _normalize_question_type(question.get("type"))
            if self.question_types and qtype not in self.question_types:
                unexpected.add(qtype)

            if image_ext is None:
                image_ext = self._image_extension(image_dir, image_id)
            seen_images.add(image_id)

            yield self.make_sample(
                f"{self.name}_{self.split}_{question_id}",
                question=question_text,
                answer=texts[0],
                references=texts[1:],
                images=(
                    self.image_ref(
                        str(image_id),
                        image_dir / f"{image_id}{image_ext}",
                        modality="optical",
                        sensor=self.sensor,
                        tile_size=self.tile_size,
                    ),
                ),
                meta={
                    "question_type": qtype,
                    "question_type_raw": question.get("type"),
                    "question_id": question_id,
                    "image_id": image_id,
                },
            )

        stats["n_images"] = len(seen_images)
        stats["unexpected_types"] = sorted(unexpected)
        if unexpected:
            log.warning(
                "%s/%s: question types outside the documented vocabulary %s: %s",
                self.name,
                self.split,
                self.question_types,
                stats["unexpected_types"],
            )
        if stats["dropped_inactive"]:
            log.info(
                "%s/%s: dropped %d inactive QA pairs (question=%d image=%d answer=%d)",
                self.name,
                self.split,
                stats["dropped_inactive"],
                stats["dropped_inactive_question"],
                stats["dropped_inactive_image"],
                stats["dropped_inactive_answer"],
            )
        self.load_stats = stats

    # -- reporting ----------------------------------------------------------- #

    def describe(self) -> dict[str, Any]:
        """Base metadata plus the parse bookkeeping, under ``meta``.

        ``load_stats`` is populated by :meth:`_load`, so ``meta`` carries only
        the static keys until the split is actually iterated -- ``satquery
        datasets`` lists availability without parsing anything.
        """
        out = super().describe()
        out["meta"] = {
            "sensor": self.sensor,
            "tile_size": self.tile_size,
            "question_types": list(self.question_types),
            "split_token": self.split_token,
            **dict(self.load_stats),
        }
        return out


# --------------------------------------------------------------------------- #
# Join helpers. Module-level and side-effect free so the two collections share
# one implementation and the test can exercise them through the public parser.
# --------------------------------------------------------------------------- #


def _kind_of(filename: str) -> str:
    """``"LR_split_test_answers.json"`` -> ``"answers"``."""
    return filename.rsplit("_", 1)[-1].removesuffix(".json")


def _index_answers(
    rows: Iterable[Any],
) -> tuple[dict[int, list[Mapping[str, Any]]], dict[int, Mapping[str, Any]]]:
    """Index the answers table both ways the two layouts need.

    ``by_question`` is the official join (``answers[i]["question_id"]``).
    ``by_id`` serves re-exports where the question row carries ``answers_ids``
    instead. Both are built in one pass; the tables are up to 86 MB for HR train.
    """
    by_question: dict[int, list[Mapping[str, Any]]] = {}
    by_id: dict[int, Mapping[str, Any]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        by_id[_row_id(row, position, "id")] = row
        if "question_id" in row:
            try:
                by_question.setdefault(int(row["question_id"]), []).append(row)
            except (TypeError, ValueError):
                continue
    return by_question, by_id


def _image_gate(rows: Iterable[Any]) -> tuple[set[int] | None, set[int]]:
    """Tile-level activity, as ``(allowed_question_ids, inactive_image_ids)``.

    ``allowed_question_ids`` is ``None`` when the table carries no
    ``questions_ids`` -- absence has to mean "no opinion", not "everything is
    inactive", or a mirror that ships a slimmed images file would silently zero
    the split.
    """
    allowed: set[int] = set()
    inactive: set[int] = set()
    saw_question_ids = False
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        active = _is_active(row)
        if not active:
            inactive.add(_row_id(row, position, "id", "img_id"))
        if "questions_ids" in row:
            saw_question_ids = True
            if active:
                allowed |= _int_set(row["questions_ids"])
    return (allowed if saw_question_ids else None), inactive


def _answer_texts(
    question: Mapping[str, Any],
    question_id: int,
    answers_by_question: Mapping[int, list[Mapping[str, Any]]],
    answers_by_id: Mapping[int, Mapping[str, Any]],
) -> tuple[tuple[str, ...], str]:
    """Every *active* answer for one question, canonical first, plus a drop reason.

    RSVQA allows several answers per question (one per annotator); in practice
    there is one. Extras become ``references`` so the accuracy metric accepts any
    of them -- that is what :attr:`Sample.all_answers` is for.
    """
    candidates: list[Mapping[str, Any]] = list(answers_by_question.get(question_id, ()))
    if not candidates:
        ids = question.get("answers_ids")
        if ids is None and "answer_id" in question:
            ids = question["answer_id"]
        for raw_id in sorted(_int_set(ids)):
            row = answers_by_id.get(raw_id)
            if row is not None:
                candidates.append(row)

    texts: list[str] = []
    for row in candidates:
        if not _is_active(row):
            continue
        text = str(row.get("answer", "")).strip()
        if text and text not in texts:
            texts.append(text)

    if not texts and question.get("answer"):
        # Some re-exports inline the answer on the question row.
        texts.append(str(question["answer"]).strip())

    if texts:
        return tuple(texts), ""
    return (), "dropped_inactive_answer" if candidates else "dropped_no_answer"


# --------------------------------------------------------------------------- #
# The two collections
# --------------------------------------------------------------------------- #


class RSVQALRDataset(_RSVQADataset):
    """RSVQA-LR: 772 Sentinel-2 tiles, 77,232 QA pairs.

    "Low resolution" means 10 m Sentinel-2 cropped to 256x256 tiles, so ``count``
    questions are genuinely hard and ``rural_urban`` is a scene-level label
    dressed up as a question -- both worth keeping as separate rows in the
    report instead of averaging them into one accuracy.
    """

    name = "rsvqa_lr"
    splits = ("train", "val", "test")
    file_prefix = "LR_split"
    split_tokens = {"train": "train", "val": "val", "test": "test"}
    image_dirs = ("Images_LR", "images_LR", "Images_LR/Images_LR", "Images", "images", "Data")
    question_types = ("presence", "count", "comparison", "rural_urban")
    sensor = "Sentinel-2"
    tile_size = 256
    homepage = "https://rsvqa.sylvainlobry.com/"
    download_hint = (
        "https://zenodo.org/records/6344334 -- LR_split_{train,val,test}_"
        "{questions,answers,images}.json plus Images_LR.zip, unzipped so that "
        "Images_LR/<img_id>.tif sits under the dataset root"
    )


class RSVQAHRDataset(_RSVQADataset):
    """RSVQA-HR: 10,659 USGS HRO aerial tiles, 1,066,316 QA pairs.

    ``test2`` is the Philadelphia set (``USGS_split_test_phili_*``), a city
    absent from training. Published RSVQA-HR numbers are usually the ``test``
    ones; the gap between the two splits is this benchmark's domain-shift
    evidence, and the hidden Cartosat/RISAT eval makes that gap the number to
    watch rather than the headline.
    """

    name = "rsvqa_hr"
    splits = ("train", "val", "test", "test2")
    file_prefix = "USGS_split"
    split_tokens = {"train": "train", "val": "val", "test": "test", "test2": "test_phili"}
    image_dirs = ("Images/Data", "Data", "Images", "images", "Data/Data")
    question_types = ("presence", "count", "comparison", "area")
    sensor = "USGS HRO aerial"
    tile_size = 512
    homepage = "https://rsvqa.sylvainlobry.com/"
    download_hint = (
        "https://zenodo.org/records/6344367 -- USGS_split_{train,val,test,test_phili}_"
        "{questions,answers,images}.json plus Images.tar (13.5 GB), untarred so that "
        "Images/Data/<img_id>.tif sits under the dataset root"
    )
