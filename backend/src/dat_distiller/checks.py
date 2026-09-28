"""Checks and Acknowledgements: responsible-use safeguards.

A **Check** is a safeguard evaluated at the end of a step (PII found,
near-copies, class imbalance, leakage, unreviewed labels, fairness gaps,
soft limits, ...). Severity is ``info`` or ``warning``. Checks never block:
a warning just needs an explicit **Acknowledgement** before the step that
raised it counts as complete. Acknowledgements are persisted so Cards can
include them.

Steps register Checks through :meth:`CheckStore.register`. The UI lists them
per subject and acknowledges warnings; ``unacknowledged_warnings`` is the
gate the API reports.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

SEVERITIES = ("info", "warning")
SUBJECT_TYPES = ("project", "dataset_version", "training_run")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checks (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('info', 'warning')),
    message TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    subject_type TEXT NOT NULL CHECK (subject_type IN ('project', 'dataset_version', 'training_run')),
    subject_id TEXT NOT NULL,
    step TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS acknowledgements (
    check_id TEXT PRIMARY KEY REFERENCES checks(id) ON DELETE CASCADE,
    acknowledged_by TEXT NOT NULL,
    note TEXT,
    acknowledged_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_checks_subject ON checks(subject_type, subject_id);
"""


class CheckNotFoundError(KeyError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Check:
    id: str
    kind: str
    severity: str
    message: str
    details: dict[str, Any]
    subject_type: str
    subject_id: str
    step: str | None
    created_at: str
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "step": self.step,
            "created_at": self.created_at,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at,
            "note": self.note,
        }


class CheckStore:
    """Checks + Acknowledgements over the same SQLite file as the data store."""

    def __init__(self, database) -> None:
        """Share the DatasetStore's SQLite `Database` (same app.db file)."""
        self.db = database
        with self.db.connect() as conn:
            conn.executescript(_SCHEMA)

    # -- registration --------------------------------------------------------

    def register(
        self,
        *,
        kind: str,
        severity: str,
        message: str,
        subject_type: str,
        subject_id: str,
        details: dict[str, Any] | None = None,
        step: str | None = None,
    ) -> Check:
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")
        if subject_type not in SUBJECT_TYPES:
            raise ValueError(f"subject_type must be one of {SUBJECT_TYPES}")
        check = Check(
            id=uuid.uuid4().hex,
            kind=kind,
            severity=severity,
            message=message,
            details=details or {},
            subject_type=subject_type,
            subject_id=subject_id,
            step=step,
            created_at=_now(),
        )
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO checks (id, kind, severity, message, details_json, "
                "subject_type, subject_id, step, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    check.id,
                    check.kind,
                    check.severity,
                    check.message,
                    json.dumps(check.details),
                    check.subject_type,
                    check.subject_id,
                    check.step,
                    check.created_at,
                ),
            )
        return check

    def has_check(self, kind: str, subject_type: str, subject_id: str) -> bool:
        """Whether a Check of this kind already exists for the subject
        (used e.g. to raise a CLI-lockdown warning only once per Project)."""
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM checks WHERE kind = ? AND subject_type = ? AND subject_id = ? LIMIT 1",
                (kind, subject_type, subject_id),
            ).fetchone()
        return row is not None

    # -- querying --------------------------------------------------------------

    def list_for_subject(
        self,
        subject_type: str,
        subject_id: str,
        *,
        include_acknowledged: bool = True,
    ) -> list[Check]:
        query = """
            SELECT c.*, a.acknowledged_by, a.note, a.acknowledged_at
            FROM checks c LEFT JOIN acknowledgements a ON a.check_id = c.id
            WHERE c.subject_type = ? AND c.subject_id = ?
        """
        if not include_acknowledged:
            query += " AND a.check_id IS NULL"
        query += " ORDER BY c.created_at, c.rowid"
        with self.db.connect() as conn:
            rows = conn.execute(query, (subject_type, subject_id)).fetchall()
        return [self._from_row(row) for row in rows]

    def unacknowledged_warnings(
        self, subject_type: str, subject_id: str
    ) -> list[Check]:
        return [
            check
            for check in self.list_for_subject(
                subject_type, subject_id, include_acknowledged=False
            )
            if check.severity == "warning"
        ]

    def get(self, check_id: str) -> Check:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT c.*, a.acknowledged_by, a.note, a.acknowledged_at
                FROM checks c LEFT JOIN acknowledgements a ON a.check_id = c.id
                WHERE c.id = ?
                """,
                (check_id,),
            ).fetchone()
        if row is None:
            raise CheckNotFoundError(check_id)
        return self._from_row(row)

    # -- acknowledging -----------------------------------------------------------

    def acknowledge(
        self, check_id: str, *, acknowledged_by: str = "local", note: str | None = None
    ) -> Check:
        """Record an Acknowledgement. Idempotent: a second call keeps the first."""
        existing = self.get(check_id)
        if existing.acknowledged:
            return existing
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO acknowledgements (check_id, acknowledged_by, note, acknowledged_at) "
                "VALUES (?, ?, ?, ?)",
                (check_id, acknowledged_by, note, _now()),
            )
        return self.get(check_id)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Check:
        return Check(
            id=row["id"],
            kind=row["kind"],
            severity=row["severity"],
            message=row["message"],
            details=json.loads(row["details_json"] or "{}"),
            subject_type=row["subject_type"],
            subject_id=row["subject_id"],
            step=row["step"],
            created_at=row["created_at"],
            acknowledged=row["acknowledged_at"] is not None,
            acknowledged_by=row["acknowledged_by"],
            acknowledged_at=row["acknowledged_at"],
            note=row["note"],
        )
