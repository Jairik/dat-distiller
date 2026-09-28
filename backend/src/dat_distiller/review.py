"""The Review Queue: low-confidence labels waiting for a human.

Labeling is Jev's answer, and Jev is not always sure. Anything below the
configured **review threshold** lands in the Review Queue, where a reviewer can
**accept** it, **override** it with a different answer, or **exclude** the row
from training entirely. Applying the decisions produces a new Dataset Version —
never an edit of the one you reviewed.

Confidence is comparable across question types, which is the whole reason the
threshold can be a single number:

- **Choice** — the top option's probability.
- **Score**  — the top level's probability.
- **Noul**   — the distance from a coin flip, ``2 * |p - 0.5|``, rescaled so a
  fully committed answer reads as 1.0 rather than 0.5. Without the rescale a
  Noul could never pass a 0.8 threshold and would be silently exempt from
  review for every setting.

Rows Jev never answered have no confidence at all. They are not in the queue —
there is nothing to accept or override — but they are counted as unreviewed, so
a partially-failed Labeling run cannot slip past the Train step looking clean.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from .store.provenance import PROVENANCE_COLUMN, Provenance
from .store.store import DatasetStore

#: How the reviewer resolved one queued label.
DECISIONS = ("accept", "override", "exclude")
Decision = Literal["accept", "override", "exclude"]

#: Provenance origin written for a human-resolved label.
REVIEWED_ORIGIN = "human_reviewed"
#: Column added to a version to mark rows a reviewer excluded.
EXCLUDED_COLUMN = "__excluded__"
#: Kind of the Check raised when labels are still sitting in the queue.
UNREVIEWED_CHECK_KIND = "unreviewed_labels"


# -- confidence ---------------------------------------------------------------


def option_column(option: str) -> str:
    """The ``__p_<option>`` column name for a Choice option (mirrors `label.py`)."""
    return "p_" + re.sub(r"[^a-zA-Z0-9_]+", "_", option).strip("_").lower()


def label_family(column: str) -> str:
    """``tone__p_pos`` -> ``tone``; a plain column is its own family."""
    return column.split("__", 1)[0]


def noul_confidence(answer_confidence: Any) -> float | None:
    """A Noul's confidence, rescaled from a coin flip to a full commitment.

    Jev states how sure it is of the answer it gave, so a "yes" at 0.9 implies
    ``P(yes) = 0.9``. Distance from 0.5 is that same information on the same
    0–1 scale the other two question types use.
    """
    value = _as_float(answer_confidence)
    if value is None:
        return None
    return max(0.0, min(1.0, 2.0 * abs(value - 0.5)))


def _as_float(value: Any) -> float | None:
    """`value` as a float, or None when it is missing or not a number."""
    if value is None or value is pd.NA:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def confidence_of(frame: pd.DataFrame, family: str, question_type: str) -> float | None:
    """One row's confidence for one Label Column family.

    Reads the siblings Labeling already wrote rather than re-asking anything:
    the top ``__p_*`` option (Choice), the top entry of ``__probabilities``
    (Score), or the rescaled stated confidence (Noul).
    """
    if question_type == "noul":
        return noul_confidence(_cell(frame, f"{family}__confidence"))
    if question_type == "choice":
        columns = [c for c in frame.columns if c.startswith(f"{family}__p_")]
        values = [v for v in (_as_float(_cell(frame, c)) for c in columns) if v is not None]
        return max(values) if values else None
    if question_type == "score":
        raw = _cell(frame, f"{family}__probabilities")
        if isinstance(raw, str) and raw:
            try:
                probabilities = json.loads(raw)
            except json.JSONDecodeError:
                return None
            values = [float(p) for p in probabilities.values()]
            finite = [v for v in values if not math.isnan(v)]
            return max(finite) if finite else None
        return _as_float(_cell(frame, f"{family}__confidence"))
    return None


def _cell(frame: pd.DataFrame, column: str) -> Any:
    """The single value of a one-row frame, or None when the column is absent."""
    if column not in frame.columns or len(frame) == 0:
        return None
    return frame[column].iloc[0]


# -- reading the queue --------------------------------------------------------


@dataclass
class QueueItem:
    """One label awaiting a decision."""

    row_index: int
    family: str
    question_type: str
    answer: Any
    confidence: float | None
    decision: str | None = None
    override: Any = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "family": self.family,
            "question_type": self.question_type,
            "answer": _jsonable(self.answer),
            "confidence": self.confidence,
            "decision": self.decision,
            "override": _jsonable(self.override),
            "note": self.note,
        }


def _jsonable(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, int, str)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


@dataclass
class Queue:
    """The Review Queue for one Dataset Version at one threshold."""

    version_id: str
    threshold: float
    families: dict[str, str] = field(default_factory=dict)
    items: list[QueueItem] = field(default_factory=list)
    #: Label Columns Jev never answered — nothing to review, but not clean.
    unlabeled: list[dict[str, Any]] = field(default_factory=list)
    #: Total items before `items` was paged. `None` until `build_queue` fills it.
    total: int | None = None

    @property
    def outstanding(self) -> int:
        """How many labels are queued across the whole version, decided or not."""
        return len(self.items) if self.total is None else self.total

    @property
    def unreviewed_count(self) -> int:
        """What the Train step reads: queued labels plus labels Jev never gave.

        Deliberately computed from `outstanding`, not from the (possibly paged)
        `items` — a truncated page must never under-report what is outstanding,
        or the Train step would read a partially-reviewed version as clean.
        """
        decided = sum(1 for item in self.items if item.decision)
        return self.outstanding - decided + len(self.unlabeled)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "threshold": self.threshold,
            "families": self.families,
            "items": [item.to_dict() for item in self.items],
            "unlabeled": self.unlabeled,
            "queued_count": self.outstanding,
            "returned_count": len(self.items),
            "unreviewed_count": self.unreviewed_count,
        }


def label_families(frame: pd.DataFrame) -> dict[str, str]:
    """The Label Columns in a version, mapped to their question type.

    Type is read off the column set Labeling produced, which is the only record
    of it that travels with the data — the questions themselves live in the
    parent version's ``meta``. The checks are ordered most-specific first, so a
    family is recognised from any one of its siblings rather than requiring all
    of them: a Choice is a Choice because it has ``__p_*`` columns, whether or
    not a ``__confidence`` column survived alongside them.
    """
    families: dict[str, str] = {}
    for column in frame.columns:
        if column == PROVENANCE_COLUMN or "__" in column:
            continue
        if any(c.startswith(f"{column}__p_") for c in frame.columns):
            families[column] = "choice"
        elif f"{column}__probabilities" in frame.columns:
            families[column] = "score"
        elif f"{column}__confidence" in frame.columns:
            families[column] = "noul"
    return families


def is_reviewed(frame: pd.DataFrame, index: int, family: str) -> bool:
    """Whether a person has already resolved this Label Column on this row.

    Without this an **accepted** label would re-enter the queue on the very next
    read — its confidence is unchanged by accepting, so the only thing that can
    keep it out is the Provenance.
    """
    if PROVENANCE_COLUMN not in frame.columns:
        return False
    provenance = Provenance.parse(frame[PROVENANCE_COLUMN].iloc[index])
    origins = provenance.get("label_origins") or {}
    entry = origins.get(family) or {}
    return entry.get("origin") == REVIEWED_ORIGIN


def build_queue(
    frame: pd.DataFrame,
    version_id: str,
    threshold: float,
    families: dict[str, str] | None = None,
    rows: int | None = None,
) -> Queue:
    """The labels on this version that are below `threshold`, or never given.

    Already-reviewed labels are left out entirely: a human has looked at them,
    and re-asking would be the queue refusing to believe an answer.

    `rows` caps the returned items (the Review Queue is paged); the counts and
    `unreviewed_count` always describe the whole version, so a capped queue can
    never under-report what is outstanding.
    """
    kinds = families if families is not None else label_families(frame)
    queue = Queue(version_id=version_id, threshold=threshold, families=kinds)
    for index in range(len(frame)):
        row = frame.iloc[[index]]
        for family, question_type in kinds.items():
            if is_reviewed(frame, index, family):
                continue
            answer = _cell(row, family)
            confidence = confidence_of(row, family, question_type)
            if confidence is None:
                if _is_missing(answer):
                    queue.unlabeled.append(
                        {"row_index": index, "family": family, "question_type": question_type}
                    )
                continue
            if confidence < threshold:
                queue.items.append(
                    QueueItem(
                        row_index=index,
                        family=family,
                        question_type=question_type,
                        answer=_jsonable(answer),
                        confidence=confidence,
                    )
                )
    if rows is not None:
        queue.total = len(queue.items)
        queue.items = queue.items[:rows]
    return queue


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


# -- applying decisions -------------------------------------------------------


@dataclass
class ReviewOutcome:
    accepted: int = 0
    overridden: int = 0
    excluded: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "overridden": self.overridden,
            "excluded_rows": sorted(self.excluded),
            "excluded_count": len(self.excluded),
        }


def validate_decisions(items: Iterable[dict[str, Any]]) -> list[QueueItem]:
    """Check a submitted decision set before anything is written.

    An override with no value, or an override of a row that was never queued,
    is a mistake worth catching here rather than half-applying.
    """
    parsed: list[QueueItem] = []
    for raw in items:
        decision = str(raw.get("decision") or "")
        if decision not in DECISIONS:
            raise ValueError(f"review decision must be one of {DECISIONS}, got {decision!r}")
        family = str(raw.get("family") or "").strip()
        if not family:
            raise ValueError("a review decision needs the Label Column it applies to")
        row_index = int(raw.get("row_index"))
        if row_index < 0:
            raise ValueError("row_index must not be negative")
        override = raw.get("override")
        if decision == "override" and _is_missing(override):
            raise ValueError(
                f"overriding {family!r} on row {row_index} needs a new answer"
            )
        parsed.append(
            QueueItem(
                row_index=row_index,
                family=family,
                question_type=str(raw.get("question_type") or ""),
                answer=raw.get("answer"),
                confidence=_as_float(raw.get("confidence")),
                decision=decision,
                override=override,
                note=(str(raw["note"]).strip() or None) if raw.get("note") else None,
            )
        )
    return parsed


def apply_decisions(
    frame: pd.DataFrame,
    decisions: list[QueueItem],
    families: dict[str, str],
) -> tuple[pd.DataFrame, ReviewOutcome]:
    """A new frame with the decisions applied. The input is never mutated.

    - **accept** — the answer stands, but the Provenance becomes
      ``human_reviewed``: a person looked at it and agreed, which is a
      materially different fact from "Jev said this and nobody checked".
    - **override** — the answer is replaced *and* every sibling probability is
      rewritten, so a Choice override cannot leave ``__p_pos`` at 0.9 next to a
      new answer of "neg". A stale sibling is exactly the kind of thing that
      leaks into training as a feature.
    - **exclude** — the row is kept (the version is a record, not a filter) but
      flagged in ``__excluded__`` so the Train step can drop it.
    """
    out = frame.copy()
    outcome = ReviewOutcome()
    excluded_rows: set[int] = set()
    by_row: dict[int, list[QueueItem]] = {}
    for item in decisions:
        by_row.setdefault(item.row_index, []).append(item)

    for row_index, items in by_row.items():
        if row_index >= len(out):
            continue
        if any(item.decision == "exclude" for item in items):
            excluded_rows.add(row_index)
            continue
        for item in items:
            if item.family not in families:
                raise ValueError(
                    f"{item.family!r} is not a Label Column on this version "
                    f"(have: {sorted(families)})"
                )
            if item.decision == "accept":
                _mark_reviewed(out, row_index, item.family, item.answer, item.note)
                outcome.accepted += 1
            else:
                _write_override(out, row_index, item, families[item.family])
                _mark_reviewed(out, row_index, item.family, item.override, item.note)
                outcome.overridden += 1

    if excluded_rows:
        flags = _excluded_flags(len(out), out.get(EXCLUDED_COLUMN), excluded_rows)
        out[EXCLUDED_COLUMN] = flags
    outcome.excluded = sorted(excluded_rows)
    return out, outcome


def _excluded_flags(
    rows: int, existing: Any, excluded: set[int]
) -> pd.Series:
    """Carry forward earlier exclusions instead of resetting them each pass."""
    flags: list[bool] = []
    for index in range(rows):
        was_excluded = False
        if existing is not None:
            value = existing.iloc[index]
            was_excluded = not _is_missing(value) and bool(value)
        flags.append(was_excluded or index in excluded)
    return pd.Series(flags, dtype="boolean")


def _mark_reviewed(
    frame: pd.DataFrame, row_index: int, family: str, answer: Any, note: str | None
) -> None:
    """Stamp `human_reviewed` Provenance on one Label Column of one row."""
    if PROVENANCE_COLUMN not in frame.columns:
        return
    provenance = Provenance.parse(frame[PROVENANCE_COLUMN].iloc[row_index])
    label_origins = dict(provenance.get("label_origins") or {})
    entry: dict[str, Any] = {"origin": REVIEWED_ORIGIN}
    previous = label_origins.get(family) or {}
    if previous.get("confidence") is not None:
        entry["jev_confidence"] = previous.get("confidence")
    if note:
        entry["note"] = note
    label_origins[family] = entry
    provenance["label_origins"] = label_origins
    frame.iat[row_index, frame.columns.get_loc(PROVENANCE_COLUMN)] = json.dumps(provenance)


def _write_override(
    frame: pd.DataFrame, row_index: int, item: QueueItem, question_type: str
) -> None:
    """Write the new answer and bring its siblings along with it."""
    family = item.family
    # A Score Label Column holds the *expected score* (a number on the
    # described scale), not the level's name — the phrasing lives only in the
    # question, which does not travel with the data. So the reviewer overrides
    # with a number. Validate before writing anything, or pandas raises a
    # dtype error that names the symptom rather than the mistake.
    level: float | None = None
    if question_type == "score":
        level = _as_float(item.override)
        if level is None:
            raise ValueError(
                f"overriding Score {family!r} on row {row_index} needs a number on the "
                f"scale, not {item.override!r}"
            )

    _set(frame, row_index, family, item.override)
    if question_type == "choice":
        # the human's answer is now certain; the old probabilities are not
        target = f"{family}__{option_column(str(item.override))}"
        for column in [c for c in frame.columns if c.startswith(f"{family}__p_")]:
            _set(frame, row_index, column, 1.0 if column == target else 0.0)
    elif question_type == "score":
        assert level is not None  # checked above
        _set(frame, row_index, f"{family}__probabilities", json.dumps({_level_key(level): 1.0}))
    # A Noul has no per-option siblings; its stated confidence follows the answer.
    _set(frame, row_index, f"{family}__confidence", 1.0)


def _level_key(score: float) -> str:
    """How a Score level is named in the probability map (5.0 reads as "5")."""
    return str(int(score)) if float(score).is_integer() else repr(float(score))


def _set(frame: pd.DataFrame, row_index: int, column: str, value: Any) -> None:
    if column not in frame.columns:
        return
    frame.iat[row_index, frame.columns.get_loc(column)] = value


# -- the store-facing operation ----------------------------------------------


def apply_review(
    store: DatasetStore,
    version_id: str,
    decisions: list[QueueItem],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Apply decisions to one Dataset Version, producing a new one.

    The source Version is immutable, so the reviewed version is a child — the
    same rule every other step follows.
    """
    version = store.get_version(version_id)
    frame = store.load_dataframe(version.id, include_provenance=True)
    families = label_families(frame)
    if not families:
        raise ValueError("this Dataset Version has no Label Columns to review")
    cleaned, outcome = apply_decisions(frame, decisions, families)

    remaining = build_queue(cleaned, version.id, threshold, families)
    child = store.create_version(
        version.project_id,
        cleaned,
        parent_id=version.id,
        origin="reviewed",
        meta={
            "review": {
                "parent_version_id": version.id,
                "threshold": threshold,
                "decisions": [item.to_dict() for item in decisions],
                "outcome": outcome.to_dict(),
                "unreviewed_count": remaining.unreviewed_count,
            }
        },
    )
    return {
        "version": child.to_dict(),
        "outcome": outcome.to_dict(),
        "unreviewed_count": remaining.unreviewed_count,
    }


def register_unreviewed_check(
    checks: Any, version_id: str, unreviewed: int, threshold: float, total: int
) -> Any | None:
    """The warning Check for labels still sitting in the queue.

    A Check rather than a block, per the design: training on unreviewed labels
    is a choice the user can make, but they should not be able to make it
    without being told how many are outstanding.
    """
    if unreviewed <= 0 or checks.has_check(UNREVIEWED_CHECK_KIND, "dataset_version", version_id):
        return None
    return checks.register(
        kind=UNREVIEWED_CHECK_KIND,
        severity="warning",
        message=(
            f"{unreviewed} label(s) on this Dataset Version are below the review threshold of "
            f"{threshold} or were never answered by Jev. Review them, or train on them knowingly."
        ),
        subject_type="dataset_version",
        subject_id=version_id,
        details={
            "unreviewed": unreviewed,
            "threshold": threshold,
            "labels": total,
        },
        step="review",
    )


__all__ = [
    "DECISIONS",
    "EXCLUDED_COLUMN",
    "REVIEWED_ORIGIN",
    "UNREVIEWED_CHECK_KIND",
    "Queue",
    "QueueItem",
    "ReviewOutcome",
    "apply_decisions",
    "apply_review",
    "build_queue",
    "confidence_of",
    "is_reviewed",
    "label_families",
    "noul_confidence",
    "register_unreviewed_check",
    "validate_decisions",
]
