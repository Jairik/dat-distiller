"""PII detection and the **warn / mask / drop** actions.

Vocabulary comes from ``CONTEXT.md``: a *PII finding* is one detector's hit
count in one column of one Dataset Version, and the *actions* you can take on a
finding are:

``warn``
    register a **warning Check** per affected column. There is deliberately no
    "warn" endpoint: the **Acknowledgement** of that Check
    (``POST /api/checks/{id}/acknowledge``) *is* the warn path, and Checks never
    block anything.
``mask``
    rewrite every detected span to ``[REDACTED]`` in a new Dataset Version.
``drop``
    remove the whole column in a new Dataset Version.

Detection is two layers, and the first one never needs a dependency:

* the **regex baseline** (:func:`scan_dataframe`) — email, US-ish phone, SSN,
  Luhn-checked card numbers, IPv4. It is the only layer used in the default
  install and by the upload / Generation hooks, because those must stay cheap.
* **Presidio** (:func:`presidio_findings`) — ``PERSON`` and ``US_ADDRESS``,
  imported lazily *inside* the function. Presidio is an optional extra
  (``pip install 'dat-distiller[presidio]'``); when it cannot be imported, or
  cannot initialise (its spaCy model is missing), it contributes nothing and the
  baseline stands alone. Nothing in this package may import it at module scope.

No function here ever hands back raw PII. Findings carry **masked** examples
(``j***.com``) and row indices only, which is what makes it safe to store them
in Check details and ship them in API responses.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .store.provenance import PROVENANCE_COLUMN

#: Kind of the Check raised for a column with findings (the `warn` action).
PII_CHECK_KIND = "pii_found"
#: Value written over a detected span by the `mask` action.
REDACTION = "[REDACTED]"
#: The two destructive actions, from CONTEXT.md's action vocabulary.
ACTIONS = ("mask", "drop")
#: How many masked examples / row indices one finding keeps.
MAX_EXAMPLE_CELLS = 3
MAX_ROW_EXAMPLES = 5

# -- helpers -----------------------------------------------------------------


def luhn(number: str) -> bool:
    """Whether a digit string passes the Luhn check (card numbers do).

    Non-digits are ignored, so a spaced or dashed candidate can be passed in
    as-is. A single digit is not a card number and reports False.
    """
    digits = [c for c in str(number) if c.isdigit()]
    if len(digits) < 2:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def mask_span(text: Any) -> str:
    """Mask one detected span: first character, ``***``, plus the email TLD.

    ``jane.doe@example.com`` -> ``j***.com``. The point is that the raw value
    can never be recovered (or grep'd for) from what we persist.
    """
    value = str(text)
    head = value[:1]
    if "@" in value:
        domain = value.rpartition("@")[2]
        tld = domain.rsplit(".", 1)[-1] if "." in domain else ""
        return f"{head}***.{tld}" if tld else f"{head}***"
    return f"{head}***"


# -- detectors ---------------------------------------------------------------

# Word/number-boundary guards everywhere: `(?<![\d.])`/`(?![\d.])` is what stops
# "version 1.212.555.0187" from yielding a phone and 999.1.1.1 / 1.2.3.4.5 from
# yielding an IP.
EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+"
    r"(?:\.[A-Za-z0-9-]+)+(?![A-Za-z0-9-])"
)
# US-ish: optional +1 (a bare "1" only counts as a country code when a space or
# dash follows, so dotted version strings cannot masquerade as numbers), then
# 3-3-4 with the usual separators, parentheses allowed — or all ten digits
# run together, which is how a number pasted out of a spreadsheet looks.
PHONE_RE = re.compile(
    r"(?<![\d.])(?:\+1[\s.-]?|1[ -])?"
    r"(?:\(?[2-9]\d{2}\)?[\s.-]\d{3}[\s.-]\d{4}|[2-9]\d{2}\d{3}\d{4})"
    r"(?![\d.])"
)
# Dashed SSN only, with the ranges the SSA never issues excluded.
SSN_RE = re.compile(r"(?<!\d)(?!000|666|9\d\d)(\d{3})-(\d{2})-(\d{4})(?!\d)")
# 13-19 digits in groups of one, spaces/dashes tolerated; Luhn decides.
CARD_RE = re.compile(r"(?<![\d.])(?:\d[ -]?){12,18}\d(?![\d.])")
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IP_RE = re.compile(rf"(?<![\d.]){_OCTET}\.{_OCTET}\.{_OCTET}\.{_OCTET}(?![\d.])")


def _is_card(candidate: str) -> bool:
    digits = re.sub(r"[ -]", "", candidate)
    return 13 <= len(digits) <= 19 and luhn(digits)


@dataclass(frozen=True)
class Detector:
    """One named pattern, an optional validator, and span-level rewriting."""

    name: str
    regex: re.Pattern[str]
    validate: Callable[[str], bool] | None = None

    def spans(self, text: str) -> list[tuple[int, int, str]]:
        """Validated ``(start, end, span)`` matches — raw spans, keep internal."""
        found = []
        for match in self.regex.finditer(text):
            span = match.group(0)
            if self.validate is None or self.validate(span):
                found.append((match.start(), match.end(), span))
        return found

    def rewrite(self, text: str, replacement: Callable[[str], str]) -> tuple[str, int]:
        """Replace every validated span; returns ``(new_text, replacements)``."""
        spans = self.spans(text)
        if not spans:
            return text, 0
        parts: list[str] = []
        cursor = 0
        for start, end, span in spans:
            parts.append(text[cursor:start])
            parts.append(replacement(span))
            cursor = end
        parts.append(text[cursor:])
        return "".join(parts), len(spans)


DETECTORS: tuple[Detector, ...] = (
    Detector("email", EMAIL_RE),
    Detector("phone", PHONE_RE),
    Detector("ssn", SSN_RE),
    Detector("credit_card", CARD_RE, _is_card),
    Detector("ip", IP_RE),
)
DETECTORS_BY_NAME = {detector.name: detector for detector in DETECTORS}


# -- scanning ----------------------------------------------------------------


def _is_text_like(series: pd.Series) -> bool:
    """Whether a column is worth scanning.

    Structured numbers are skipped: PII that lives in a text column is the case
    that actually leaks into Cards and Provider prompts, and stringifying every
    numeric column of a large frame would make the upload hook expensive.
    """
    return bool(
        pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _row_index(index: Any) -> Any:
    try:
        return int(index)
    except (TypeError, ValueError):
        return str(index)


def _cell_text(value: Any) -> str | None:
    """The string to scan for one cell, or None when there is nothing to scan.

    Missing values are skipped rather than stringified: `pd.NA`, `None` and
    `NaN` all become "nan"/"<NA>" and would be noise in a finding.
    """
    if value is None or value is pd.NA:
        return None
    if not isinstance(value, str) and pd.isna(value):
        return None
    return value if isinstance(value, str) else str(value)


def _scan_column(column: str, series: pd.Series, spans_for) -> list[dict[str, Any]]:
    """Build one finding per detector for one column, with masked examples."""
    counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    rows: dict[str, list[Any]] = {}
    for index, value in series.items():
        text = _cell_text(value)
        if text is None:
            continue
        for detector in DETECTORS:
            spans = spans_for(detector, text)
            if not spans:
                continue
            masked = text
            for start, end, _span in reversed(spans):
                masked = masked[:start] + mask_span(masked[start:end]) + masked[end:]
            counts[detector.name] = counts.get(detector.name, 0) + len(spans)
            if len(examples.setdefault(detector.name, [])) < MAX_EXAMPLE_CELLS:
                examples[detector.name].append(masked)
            if len(rows.setdefault(detector.name, [])) < MAX_ROW_EXAMPLES:
                rows[detector.name].append(_row_index(index))
    return [
        {
            "column": column,
            "detector": detector.name,
            "count": counts[detector.name],
            "example_cells": examples.get(detector.name, []),
            "row_examples": rows.get(detector.name, []),
        }
        for detector in DETECTORS
        if counts.get(detector.name)
    ]


def scan_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Findings for every text column of `df`, using the regex baseline only.

    ``__provenance__`` is bookkeeping, not user data, so it is never scanned (and
    never appears in a finding).
    """
    findings: list[dict[str, Any]] = []
    for column in df.columns:
        if column == PROVENANCE_COLUMN:
            continue
        series = df[column]
        if not _is_text_like(series):
            continue
        findings.extend(_scan_column(str(column), series, lambda d, t: d.spans(t)))
    return findings


# -- Presidio (optional extra) -------------------------------------------------

#: Presidio entity types we treat as PII.
PRESIDIO_ENTITIES = ("PERSON", "US_ADDRESS")

_UNSET = object()
_ENGINE: Any = _UNSET


def _presidio_engine() -> Any:
    """The cached Presidio `AnalyzerEngine`, or None when unusable.

    Imported by name at call time — never at module scope — so the package
    imports (and the whole test suite runs) without Presidio. Anything that
    goes wrong while importing or building the engine, including a missing
    spaCy model, is treated as "Presidio is not available here".
    """
    global _ENGINE
    if _ENGINE is _UNSET:
        try:
            module = importlib.import_module("presidio_analyzer")
            engine = module.AnalyzerEngine()
        except Exception:  # noqa: BLE001 - absent, or present but unbuildable
            engine = None
        _ENGINE = engine
    return _ENGINE


def _reset_presidio_engine() -> None:
    """Drop the cached engine. Test hook: monkeypatching ``sys.modules``
    between tests must not be polluted by an earlier import."""
    global _ENGINE
    _ENGINE = _UNSET


def presidio_findings(df: pd.DataFrame) -> list[dict[str, Any]]:
    """PERSON / US_ADDRESS findings via Presidio, [] when it is unavailable.

    Same finding shape as :func:`scan_dataframe`, detector named
    ``presidio:<ENTITY>``, spans masked the same way.
    """
    engine = _presidio_engine()
    if engine is None:
        return []
    findings: list[dict[str, Any]] = []
    for column in df.columns:
        if column == PROVENANCE_COLUMN:
            continue
        series = df[column]
        if not _is_text_like(series):
            continue
        findings.extend(_presidio_column(str(column), series, engine))
    return findings


def _presidio_spans(engine: Any, text: str) -> list[tuple[str, int, int]]:
    """``(entity, start, end)`` for one cell; [] if Presidio cannot read it."""
    try:
        results = engine.analyze(text, entities=list(PRESIDIO_ENTITIES), language="en")
    except Exception:  # noqa: BLE001 - one bad cell must not stop a scan
        return []
    return [
        (str(result.entity_type), int(result.start), int(result.end))
        for result in results
        if int(result.end) > int(result.start)
    ]


def _presidio_column(column: str, series: pd.Series, engine: Any) -> list[dict[str, Any]]:
    """One finding per Presidio entity type present in this column."""
    counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    rows: dict[str, list[Any]] = {}
    for index, value in series.items():
        text = _cell_text(value)
        if text is None:
            continue
        spans = _presidio_spans(engine, text)
        if not spans:
            continue
        # attribute each span to the entity Presidio reported, not to its
        # position, so one cell can contribute to several findings at once
        by_entity: dict[str, list[tuple[int, int, str]]] = {}
        for entity, start, end in spans:
            by_entity.setdefault(entity, []).append((start, end, text[start:end]))
        for entity, hits in by_entity.items():
            key = f"presidio:{entity}"
            masked = text
            for start, end, _span in reversed(hits):
                masked = masked[:start] + mask_span(masked[start:end]) + masked[end:]
            counts[key] = counts.get(key, 0) + len(hits)
            if len(examples.setdefault(key, [])) < MAX_EXAMPLE_CELLS:
                examples[key].append(masked)
            if len(rows.setdefault(key, [])) < MAX_ROW_EXAMPLES:
                rows[key].append(_row_index(index))
    return [
        {
            "column": column,
            "detector": key,
            "count": counts[key],
            "example_cells": examples.get(key, []),
            "row_examples": rows.get(key, []),
        }
        for key in counts
    ]


def scan(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Everything this install can find: regex baseline plus Presidio if usable."""
    return scan_dataframe(df) + presidio_findings(df)


# -- actions -----------------------------------------------------------------


def apply_actions(
    df: pd.DataFrame, actions: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the destructive actions (``mask`` / ``drop``); never edits in place.

    Provenance is untouched: ``__provenance__`` is neither scannable nor a
    legitimate action target. An action for a column with no findings is a
    no-op that is still reported, so the caller can show what was requested.
    """
    out = df.copy()
    columns: dict[str, dict[str, Any]] = {}
    for column, action in (actions or {}).items():
        if action not in ACTIONS:
            raise ValueError(f"pii action must be one of {ACTIONS}, got {action!r}")
        entry: dict[str, Any] = {"action": action, "replacements": 0}
        if column == PROVENANCE_COLUMN:
            entry["note"] = "provenance column is protected"
        elif column not in out.columns:
            entry["note"] = "column not present"
        elif action == "drop":
            out = out.drop(columns=[column])
            entry["note"] = "column removed"
        else:
            series = out[column]
            replaced_total = 0
            if _is_text_like(series):
                masked_column = series.copy()
                for index, value in series.items():
                    if not isinstance(value, str):
                        continue
                    text = value
                    cell_total = 0
                    for detector in DETECTORS:
                        text, replaced = detector.rewrite(text, lambda _span: REDACTION)
                        cell_total += replaced
                    replaced_total += cell_total
                    # keep the *original* cell when nothing matched, so an
                    # untouched value is never re-boxed into a new object
                    masked_column.at[index] = text if cell_total else value
                out[column] = masked_column
            entry["replacements"] = replaced_total
            entry["note"] = (
                f"{replaced_total} span(s) redacted"
                if replaced_total
                else "no findings in this column"
            )
        columns[str(column)] = entry
    summary = {
        "columns": columns,
        "masked": [c for c, e in columns.items() if e["action"] == "mask"],
        "dropped": [c for c, e in columns.items() if e["action"] == "drop"],
        "noops": [c for c, e in columns.items() if not e["replacements"]],
    }
    return out, summary


# -- Checks (the `warn` action) ------------------------------------------------


def summarize_findings(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact form for API response bodies and job results."""
    detectors: dict[str, int] = {}
    columns: set[str] = set()
    total = 0
    for finding in findings:
        columns.add(finding["column"])
        detectors[finding["detector"]] = detectors.get(finding["detector"], 0) + finding["count"]
        total += finding["count"]
    return {
        "columns": sorted(columns),
        "detectors": detectors,
        "total_findings": total,
    }


def registered_pii_columns(checks: Any, version_id: str) -> set[str]:
    """Columns that already carry a `pii_found` Check for this Dataset Version."""
    return {
        check.details.get("column")
        for check in checks.list_for_subject("dataset_version", version_id)
        if check.kind == PII_CHECK_KIND
    }


def register_pii_checks(
    checks: Any, version_id: str, findings: list[dict[str, Any]], *, step: str = "pii_scan"
) -> list[Any]:
    """The ``warn`` action: one warning Check per affected column, deduplicated.

    A column that already has a `pii_found` Check is skipped, so re-scanning the
    same version (or scanning it again after an upload retry) cannot bury the UI
    in duplicates. `has_check` is subject-wide and cannot express "this column
    already reported", hence the per-column lookup.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for finding in findings:
        column = finding["column"]
        entry = grouped.setdefault(
            column,
            {"detectors": {}, "count": 0, "examples": [], "row_examples": []},
        )
        entry["detectors"][finding["detector"]] = (
            entry["detectors"].get(finding["detector"], 0) + finding["count"]
        )
        entry["count"] += finding["count"]
        entry["examples"].extend(finding.get("example_cells") or [])
        entry["row_examples"].extend(finding.get("row_examples") or [])
    seen = registered_pii_columns(checks, version_id)
    registered = []
    for column, entry in grouped.items():
        if column in seen:
            continue
        detectors = entry["detectors"]
        primary = max(detectors, key=lambda name: detectors[name])
        check = checks.register(
            kind=PII_CHECK_KIND,
            severity="warning",
            message=(
                f"Possible PII in column '{column}' ({detectors[primary]} hit(s) of "
                f"{primary}, {len(detectors)} detector(s))"
            ),
            subject_type="dataset_version",
            subject_id=version_id,
            details={
                "column": column,
                "detector": primary,
                "detectors": detectors,
                "count": entry["count"],
                "examples": entry["examples"][:MAX_EXAMPLE_CELLS],
                "row_examples": sorted(set(entry["row_examples"]))[:MAX_ROW_EXAMPLES],
            },
            step=step,
        )
        registered.append(check)
        seen.add(column)
    return registered


__all__ = [
    "ACTIONS",
    "DETECTORS",
    "PII_CHECK_KIND",
    "PRESIDIO_ENTITIES",
    "REDACTION",
    "apply_actions",
    "luhn",
    "mask_span",
    "presidio_findings",
    "register_pii_checks",
    "registered_pii_columns",
    "scan",
    "scan_dataframe",
    "summarize_findings",
]
