"""Shared text normalisation. Scoring-critical, therefore centralised.

Two different things live here and must not be confused:

* :func:`extract_answer` -- *prediction-side* cleanup. VLMs are chatty
  ("The answer is: two houses."). Benchmarks score exact match against a short
  string, so an un-stripped answer scores 0 for formatting reasons and makes a
  model look worse than it is. Every VQA predictor should run its raw decode
  through this before returning.
* :func:`normalize_answer` -- *comparison-side* canonicalisation applied to
  BOTH sides by the accuracy metric. Never apply it before storing ground truth;
  the raw strings stay in the sample so the report can show them verbatim.

:func:`tokenize_caption` is the captioning-metric tokeniser (BLEU / ROUGE-L /
CIDEr-D). It approximates coco-caption's PTBTokenizer without needing Java: it
lowercases, drops punctuation and splits on whitespace. That is exactly what
PTBTokenizer does for the sentence shapes that appear in RS captions (no
contractions to speak of), so scores are comparable to published numbers -- but
the caveat is documented in the report so nobody over-claims a 0.1 CIDEr delta.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

__all__ = [
    "normalize_answer",
    "extract_answer",
    "tokenize_caption",
    "ngrams",
    "is_yes_no",
    "YES_WORDS",
    "NO_WORDS",
    "NUMBER_WORDS",
]

YES_WORDS: frozenset[str] = frozenset({"yes", "yeah", "yep", "true", "correct", "affirmative"})
NO_WORDS: frozenset[str] = frozenset({"no", "nope", "false", "incorrect", "negative", "none"})

NUMBER_WORDS: dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20",
}

_ARTICLES = frozenset({"a", "an", "the"})

# Chatty VLM preambles, longest first so the longer match wins.
_PREFIXES = (
    "the answer to the question is",
    "based on the image the answer is",
    "based on the image,",
    "in the image, there is",
    "in the image, there are",
    "the answer is:",
    "the answer is",
    "answer:",
    "answer is",
    "a:",
    "response:",
    "assistant:",
    "there are",
    "there is",
    "it is",
    "this is",
)

_PUNCT_RE = re.compile(r"[^\w\s%<>-]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def extract_answer(text: str | None, *, max_words: int = 12) -> str:
    """Reduce a free-form model reply to the short span a benchmark scores.

    Deterministic and conservative: it strips known preambles, keeps the first
    sentence/line, drops trailing punctuation, and gives up (returns the first
    ``max_words`` words) rather than guessing cleverly.
    """
    if not text:
        return ""
    s = text.strip()
    # Keep the first non-empty line, then the first sentence of it.
    for line in s.splitlines():
        if line.strip():
            s = line.strip()
            break
    lowered = s.lower()
    for p in _PREFIXES:
        if lowered.startswith(p):
            s = s[len(p):].strip()
            lowered = s.lower()
            break
    s = re.split(r"(?<=[.!?])\s", s, maxsplit=1)[0]
    s = s.strip().strip('"\'`').rstrip(".!?,;:").strip()
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words])
    return s


def normalize_answer(
    text: str | None,
    *,
    map_numbers: bool = True,
    drop_articles: bool = True,
    unify_yes_no: bool = True,
) -> str:
    """Canonicalise an answer string for exact-match comparison.

    Applied to prediction *and* ground truth by the accuracy metric. Lowercases,
    strips accents and punctuation (keeping ``%``, ``-``, ``<`` and ``>`` because
    RSVQA answer classes look like ``"0m2 - 10m2"`` and ``"between 10 and 100"``),
    collapses whitespace, optionally folds articles, yes/no synonyms and English
    number words up to twenty.
    """
    if not text:
        return ""
    s = _strip_accents(str(text)).lower().strip()
    s = s.replace("_", " ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return ""
    tokens = s.split()
    if drop_articles:
        tokens = [t for t in tokens if t not in _ARTICLES] or tokens
    if map_numbers:
        tokens = [NUMBER_WORDS.get(t, t) for t in tokens]
    if unify_yes_no and len(tokens) == 1:
        if tokens[0] in YES_WORDS:
            tokens = ["yes"]
        elif tokens[0] in NO_WORDS:
            tokens = ["no"]
    return " ".join(tokens)


def is_yes_no(text: str | None) -> bool:
    """True when the (normalised) string is a yes/no answer."""
    return normalize_answer(text) in ("yes", "no")


def tokenize_caption(text: str | None) -> list[str]:
    """PTB-ish caption tokeniser: lowercase, drop punctuation, split on space."""
    if not text:
        return []
    s = _strip_accents(str(text)).lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return _WS_RE.sub(" ", s).strip().split()


def ngrams(tokens: Sequence[str], n: int) -> Iterable[tuple[str, ...]]:
    """Yield the ``n``-grams of ``tokens`` (empty when too short)."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])
