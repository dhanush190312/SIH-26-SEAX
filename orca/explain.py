"""Evidence-linked explanation: templating, verification, optional LLM narration.

The no-hallucination rule is enforced in code, in three layers:

1. **Templates.** The default answer is assembled from :mod:`orca.i18n` templates whose
   slots are filled *only* with values read from schema objects / decision results.
2. **Verifier.** Every answer -- template or model-written -- is checked by
   :func:`verify_text` against an :class:`EvidenceLedger` built from the structured
   result. A number-with-unit that is not in the ledger, any percentage, and any
   assurance-of-safety wording is rejected. (This is SatQuery's "an unbacked number
   never survives" composer rule, transplanted.)
3. **Downgrade.** :func:`gate_narration` refuses a narration that is more optimistic than
   the deterministic decision (e.g. "safe to go" when the state is INSUFFICIENT_DATA),
   and the pipeline falls back to the template text, recording why in the trace.

The optional LLM (``ORCA_LLM=mistral``) receives only ``USER QUERY + STRUCTURED EVIDENCE +
DECISION RESULT + SOURCE METADATA`` and may only rephrase. It is never asked for a fact.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol

__all__ = ["fmt", "EvidenceLedger", "verify_text", "gate_narration", "Narrator", "MistralNarrator", "build_ledger", "Verification"]


def fmt(x: float | int | None, digits: int = 2) -> str:
    """Stable number formatting: 1.40 -> '1.4', 55.0 -> '55', None -> '?'."""
    if x is None:
        return "?"
    if isinstance(x, int):
        return str(x)
    s = f"{x:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


# a number immediately followed by a measurement unit
_NUM_UNIT = re.compile(r"(?<![\w./])([-+]?\d+(?:\.\d+)?)\s*(mm/h|km/h|mg/m³|°C|m/s|km|%|h|m)(?![\w/³])")
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s*%")
_ASSURANCE = re.compile(r"\b(guarantee[sd]?|100\s*%|completely safe|totally safe|no risk|risk[- ]free|absolutely safe|safest)\b", re.I)
_GO_WORDS = re.compile(r"\b(safe to go|go ahead|you can go|it is safe|is safe|are safe|all clear|no danger|good to go)\b", re.I)
_TA_SAFE = ("பாதுகாப்பாக செல்லலாம்", "பாதுகாப்பானது")


@dataclass
class EvidenceLedger:
    """Every number the answer is allowed to mention, keyed by unit."""

    numbers: dict[str, list[float]] = field(default_factory=dict)

    def add(self, value: Any, unit: str | None = None) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        self.numbers.setdefault(unit or "*", []).append(float(value))
        self.numbers.setdefault("*", []).append(float(value))

    def allows(self, value: float, unit: str) -> bool:
        pool = self.numbers.get(unit, [])
        tol = 0.051 if unit != "km" else 0.11
        return any(abs(value - v) <= tol for v in pool)


def _walk(obj: Any, ledger: EvidenceLedger, unit_hint: str | None = None) -> None:
    if isinstance(obj, Mapping):
        unit = obj.get("unit") if isinstance(obj.get("unit"), str) else unit_hint
        for k, v in obj.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                u = unit
                if k.endswith("_km") or k in ("distance_km", "offset_km", "radius_km", "detour_km", "direct_distance_km"):
                    u = "km"
                elif k.endswith("_h") or k in ("duration_h", "freshness_hours"):
                    u = "h"
                elif k.endswith("_m") or k in ("max_wave_m",):
                    u = "m"
                elif k.endswith("_kmh") or k == "max_wind_kmh":
                    u = "km/h"
                ledger.add(v, u)
            else:
                _walk(v, ledger, unit)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, ledger, unit_hint)


def build_ledger(*structures: Any) -> EvidenceLedger:
    """Collect allowed numbers from every structured object the answer may draw on.

    Unit inference: values inherit the unit of their enclosing observation-like mapping; a
    handful of well-known key suffixes (``*_km``, ``*_h``, ``*_m``) map to km / h / m. Any
    number is also allowed with the wildcard unit ``*`` only for the *value*, never a
    bare unit match -- so 2.1 m cannot be justified by a ``2.1`` that belongs to something else
    unless it sits in the ledger under a compatible unit."""
    ledger = EvidenceLedger()
    for s in structures:
        _walk(s, ledger)
    # thresholds are stored unit-less in Thresholds.to_dict(); allow them under the obvious units
    for s in structures:
        if isinstance(s, Mapping) and "thresholds" in s:
            th = s["thresholds"]
            for k, unit in (("wave_moderate_m", "m"), ("wave_higher_m", "m"), ("wind_moderate_kmh", "km/h"),
                            ("wind_higher_kmh", "km/h"), ("precip_moderate_mmh", "mm/h"), ("chl_min_mg_m3", "mg/m³"),
                            ("sst_gradient_min_c_per_10km", "°C"), ("sst_min_c", "°C"), ("sst_max_c", "°C")):
                if k in th:
                    ledger.add(th[k], unit)
    return ledger


@dataclass(frozen=True)
class Verification:
    ok: bool
    unsupported: tuple[str, ...]
    percentages: tuple[str, ...]
    assurances: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "unsupported_numbers": list(self.unsupported), "percentages": list(self.percentages),
                "assurance_words": list(self.assurances)}


def verify_text(text: str, ledger: EvidenceLedger) -> Verification:
    bad: list[str] = []
    for m in _NUM_UNIT.finditer(text):
        val, unit = float(m.group(1)), m.group(2)
        if unit == "%":
            continue  # handled below: percentages are never allowed
        if not ledger.allows(val, unit):
            bad.append(m.group(0))
    pct = [m.group(0) for m in _PERCENT.finditer(text)]
    assur = [m.group(0) for m in _ASSURANCE.finditer(text)]
    return Verification(not (bad or pct or assur), tuple(bad), tuple(pct), tuple(assur))


def gate_narration(text: str, state: str, ledger: EvidenceLedger) -> tuple[bool, str, Verification]:
    """Accept a narration only if it verifies AND is not rosier than the decision."""
    v = verify_text(text, ledger)
    if not v.ok:
        return False, "contains unsupported numbers, percentages or assurance wording", v
    if state != "LOWER_RISK" and (_GO_WORDS.search(text) or any(w in text for w in _TA_SAFE)):
        return False, f"claims safety while the deterministic state is {state}", v
    return True, "", v


class Narrator(Protocol):
    name: str

    def narrate(self, query: str, language: str, evidence: Mapping[str, Any], decision: Mapping[str, Any], draft: str) -> str: ...


class MistralNarrator:
    """Optional rephraser (Mistral chat completions, free tier). Needs ``MISTRAL_API_KEY``.

    UNTESTED against the live API in the build environment (no network) -- it is behind
    the verifier, so a bad or missing reply just means the template answer is used."""

    name = "mistral"
    URL = "https://api.mistral.ai/v1/chat/completions"

    def __init__(self, model: str = "mistral-small-latest", key: str | None = None, timeout: float = 15.0) -> None:
        self.model, self.timeout = model, timeout
        self._key = key or os.environ.get("MISTRAL_API_KEY", "")

    @property
    def configured(self) -> bool:
        return bool(self._key)

    def narrate(self, query: str, language: str, evidence: Mapping[str, Any], decision: Mapping[str, Any], draft: str) -> str:
        system = ("You rewrite a marine decision-support answer for a fisherman in the requested language. "
                  "Use ONLY the facts in DRAFT, EVIDENCE and DECISION. Never add a number, place, forecast, warning or boundary "
                  "that is not there. Never say it is safe or guarantee anything. Keep every number and unit exactly as given "
                  "(Latin digits). Keep the decision state wording. Max 90 words.")
        payload = {"model": self.model, "temperature": 0.2, "max_tokens": 300, "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({"USER_QUERY": query, "LANGUAGE": language, "DRAFT": draft,
                                                    "DECISION": decision, "EVIDENCE": evidence}, ensure_ascii=False)[:9000]}]}
        req = urllib.request.Request(self.URL, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 - fixed https host
            return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()
