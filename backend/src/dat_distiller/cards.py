"""Cards: an exportable summary of how a Dataset Version or a Model was made.

A Card is a **record of what happened**, not marketing copy. It states what was
generated, what was labeled, what a human changed, what a Model scored, and —
repeatedly and without hedging — that Provider-written rows cannot be
reproduced exactly. A Card that read like a result without its caveats would be
worse than no Card at all.

Two hard rules, both tested:

* **No API keys, ever.** A Card is designed to be shared, and a Card that leaked
  a key would leak it to everyone it reached. Every value is filtered through
  :func:`_redact` on the way in, so a key cannot hide in a nested dict either.
* **Reproducibility is stated honestly.** A seed reproduces the *sampling*, not
  the text. Generation Modes that call a Provider say so in the Card, in the
  Dataset Card's own words, every time.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

#: The sentence every Card that involves a Provider ends on.
REPRODUCIBILITY_CAVEAT = (
    "Rows written by a Provider cannot be reproduced exactly. The recorded seed "
    "reproduces the sampling — which rows were chosen, the fitted distributions "
    "and the split — but re-running will produce different wording. Treat this "
    "Version as a snapshot, not a recipe."
)

#: Anything that looks like a credential, redacted wherever it appears.
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bsk-or-v1-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{12,}", re.IGNORECASE),
    re.compile(r"\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*\S{6,}", re.IGNORECASE),
)

REDACTED = "[REDACTED]"

#: Key names whose *values* are never safe to print, whatever they contain.
_SECRET_KEYS = re.compile(
    r"(api[_-]?key|secret|token|password|authorization|credential)", re.IGNORECASE
)


def _redact(value: Any) -> Any:
    """Recursively strip anything credential-shaped from a value.

    Applied to every value that reaches a Card, so a key cannot hide in a nested
    dict, a list, or a string that merely contains one.
    """
    if isinstance(value, Mapping):
        return {
            str(k): (REDACTED if _SECRET_KEYS.search(str(k)) else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    if isinstance(value, str):
        out = value
        for pattern in _SECRET_PATTERNS:
            out = pattern.sub(REDACTED, out)
        return out
    return value


def _is_secretish(key: str) -> bool:
    return bool(_SECRET_KEYS.search(key))


# -- Markdown rendering -------------------------------------------------------


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["_(none)_", ""]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    out.append("")
    return out


def _section(title: str, lines: list[str]) -> list[str]:
    if not any(line.strip() for line in lines):
        return []
    return [f"## {title}", "", *lines]


def _bullets(items: Any) -> list[str]:
    if not items:
        return ["_(none)_", ""]
    return [f"- {item}" for item in items] + [""]


# -- shared pieces ------------------------------------------------------------


def _checks_section(checks: list[Mapping[str, Any]], title: str = "Checks and Acknowledgements") -> list[str]:
    """Every Check and every Acknowledgement, because a Card that omits the
    acknowledged ones is hiding the human decisions that shaped the data."""
    if not checks:
        return _section(title, ["No Checks were raised."])
    rows: list[list[Any]] = []
    for check in checks:
        details = check.get("details") or {}
        note = check.get("note")
        rows.append(
            [
                f"`{check.get('kind')}`",
                check.get("severity"),
                # the message is free text written by whichever step raised the
                # Check, so it is redacted like any other value — a key quoted in
                # a message is still a key
                _redact(check.get("message")),
                "yes" if check.get("acknowledged") else "**no**",
                (check.get("acknowledged_at") or "")[:19] or "—",
                f"`{_redact(details)}`" if details else "—",
            ]
        )
        # the note belongs to *its* Check, not to whichever one happened to be
        # last in the list
        if note:
            rows.append(["", "", f"_note: {_redact(note)}_", "", "", ""])
    return _section(
        title,
        _table(
            ["Check", "Severity", "Message", "Acknowledged", "When", "Details"], rows
        )
        + [f"_{len(rows)} Check(s); {sum(1 for c in checks if c.get('acknowledged'))} acknowledged._", ""],
    )


def _library_versions() -> dict[str, str]:
    import platform
    import sys
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {"python": platform.python_version(), "platform": sys.platform}
    for name in ("pandas", "numpy", "scikit-learn", "lightgbm"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            # an optional extra this install does not have; say nothing rather
            # than print a misleading "None"
            continue
    return versions


# -- the Dataset Card ---------------------------------------------------------


@dataclass
class DatasetCard:
    version_id: str
    project_id: str
    number: int
    origin: str
    row_count: int
    columns: list[dict[str, Any]] = field(default_factory=list)
    provenance_summary: dict[str, int] = field(default_factory=dict)
    seed: int | None = None
    created_at: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    review: dict[str, Any] = field(default_factory=dict)
    pii: dict[str, Any] = field(default_factory=dict)
    parent: dict[str, Any] | None = None
    fairness: dict[str, Any] | None = None

    # -- markdown -----------------------------------------------------------

    def to_markdown(self) -> str:
        meta = self.meta if isinstance(self.meta, Mapping) else {}
        labeling = meta.get("labeling") or {}
        generation = meta.get("generation") or {}
        pii_actions = meta.get("pii") or {}
        fidelity = meta.get("fidelity") or {}

        lines: list[str] = [
            f"# Dataset Card — v{self.number}",
            "",
            f"- **Dataset Version**: `{self.version_id}`",
            f"- **Project**: `{self.project_id}`",
            f"- **Origin**: {self.origin}",
            f"- **Rows**: {self.row_count:,}",
            f"- **Created**: {self.created_at}",
            f"- **Seed**: {self.seed if self.seed is not None else 'not recorded'}",
        ]
        if self.parent:
            lines.append(f"- **Parent**: `{(self.parent or {}).get('id')}` (v{(self.parent or {}).get('number')})")
        lines.append("")

        lines += _section(
            "Provenance",
            _table(
                ["Where the rows came from", "Rows"],
                [[k, f"{v:,}"] for k, v in self.provenance_summary.items()],
            ),
        )

        lines += _section(
            "Columns",
            _table(
                ["Column", "Kind"],
                [[c.get("name"), c.get("kind")] for c in self.columns],
            ),
        )

        if generation:
            lines += _section(
                "Generation",
                _table(
                    ["Setting", "Value"],
                    [[k, _redact(v)] for k, v in generation.items()],
                ),
            )
        if fidelity:
            lines += _section(
                "Fidelity Report",
                _table(
                    ["Measure", "Value"],
                    [
                        ["Correlation drift", fidelity.get("correlation_drift")],
                        ["Near-copies", (fidelity.get("near_copies") or {}).get("count")],
                        ["Exact duplicates", (fidelity.get("exact_duplicates") or {}).get("count")],
                        ["Dropped rows", fidelity.get("dropped_rows")],
                        [
                            "Profile Provider-derived",
                            "yes — every comparison below is against a description, not reality"
                            if fidelity.get("provider_derived_profile")
                            else "no",
                        ],
                    ],
                ),
            )

        if labeling.get("questions"):
            lines += _section(
                "Jev Questions",
                _table(
                    ["Name", "Type", "Instructions", "Options / levels"],
                    [
                        [
                            f"`{q.get('name')}`",
                            q.get("type"),
                            q.get("instructions"),
                            ", ".join(str(x) for x in (q.get("criteria") or q.get("levels") or {}))
                            or "—",
                        ]
                        for q in labeling["questions"]
                    ],
                ),
            )
            if labeling.get("state_columns"):
                lines.append(f"_State columns: {', '.join(f'`{c}`' for c in labeling['state_columns'])}_")
                lines.append("")

        if self.review:
            lines += _section("Review", _table(["Measure", "Value"], [[k, v] for k, v in self.review.items()]))

        if self.pii or pii_actions:
            body = {**(self.pii or {}), **({"actions": pii_actions} if pii_actions else {})}
            lines += _section("PII handling", _table(["Key", "Value"], [[k, _redact(v)] for k, v in body.items()]))

        if self.fairness:
            lines += _section("Fairness", _table(["Key", "Value"], [[k, _redact(v)] for k, v in self.fairness.items()]))

        lines += _checks_section(self.checks)

        lines += _section(
            "Known limitations",
            _bullets(
                [
                    REPRODUCIBILITY_CAVEAT if self.origin in {"generated", "labeled"} or generation else None,
                    "Rows Jev could not answer carry null Label Columns and are not usable as a Target.",
                    "The Review Queue reflects one reviewer's judgements, not an inter-annotator agreement.",
                    f"Built on {_library_versions()['python']} with " + ", ".join(
                        f"{k} {v}" for k, v in _library_versions().items() if k not in {"python"}
                    ) + ".",
                ]
            ),
        )
        return "\n".join(lines).rstrip() + "\n"

    def to_dict(self) -> dict[str, Any]:
        """The same content as JSON, with the same redaction applied."""
        return _redact(
            {
                "kind": "dataset_card",
                "version_id": self.version_id,
                "project_id": self.project_id,
                "number": self.number,
                "origin": self.origin,
                "row_count": self.row_count,
                "columns": self.columns,
                "provenance_summary": self.provenance_summary,
                "seed": self.seed,
                "created_at": self.created_at,
                "meta": self.meta,
                "parent": self.parent,
                "review": self.review,
                "pii": self.pii,
                "fairness": self.fairness,
                "checks": self.checks,
                "reproducibility": REPRODUCIBILITY_CAVEAT,
                "library_versions": _library_versions(),
            }
        )


# -- the Model Card -----------------------------------------------------------


@dataclass
class ModelCard:
    run: dict[str, Any]
    checks: list[dict[str, Any]] = field(default_factory=list)
    fairness: dict[str, Any] | None = None
    dataset_card: dict[str, Any] | None = None

    @property
    def run_id(self) -> str:
        return str(self.run.get("training_run_id") or "")

    def to_markdown(self) -> str:
        setup = self.run.get("setup") or {}
        leaderboard = self.run.get("leaderboard") or []
        ranked = [e for e in leaderboard if e.get("rank") is not None]

        lines: list[str] = [
            f"# Model Card — Training Run `{self.run_id}`",
            "",
            f"- **Dataset Version**: `{self.run.get('version_id')}`",
            f"- **Target**: `{_redact(setup.get('target'))}`",
            f"- **Task Type**: {setup.get('task_type')}",
            f"- **Seed**: {self.run.get('seed')}",
            f"- **Primary metric**: {self.run.get('primary_metric')}"
            + (" (higher is better)" if self.run.get("primary_metric_higher_is_better") else " (lower is better)"),
            "",
        ]

        lines += _section(
            "Features",
            _table(
                ["Column", "Role"],
                [
                    [f"`{name}`", (self.run.get("preprocessing") or {}).get("roles", {}).get(name, "—")]
                    for name in (self.run.get("feature_names") or [])
                ]
                + [[f"**excluded: {k}**", v] for k, v in (setup.get("excluded_columns") or {}).items()],
            ),
        )
        sensitive = setup.get("sensitive_attribute")
        if sensitive:
            lines += _section(
                "Sensitive Attribute",
                [
                    f"`{sensitive}` — "
                    + (
                        "excluded from the features by default"
                        if not setup.get("include_sensitive_attribute")
                        else "**included** in the features by explicit opt-in"
                    ),
                    "",
                ],
            )

        lines += _section(
            "Split",
            _table(
                ["Side", "Rows"],
                [
                    ["train", (self.run.get("training_split") or {}).get("rows")],
                    ["test", (self.run.get("test_split") or {}).get("rows")],
                ],
            )
            + [
                (
                    f"_Held out at {(self.run.get('test_split') or {}).get('test_size')}, "
                    f"{'stratified' if (self.run.get('test_split') or {}).get('stratified') else 'not stratified'}. "
                    "Tuning cross-validates on the training split only._"
                ),
                "",
            ],
        )

        lines += _section(
            "Leaderboard",
            _table(
                ["Rank", "Model", "Library", "Primary", "Metrics"],
                [
                    [
                        e.get("rank"),
                        f"{e.get('label') or e.get('model')}",
                        e.get("library"),
                        _fmt((e.get("primary") or {}).get("value")),
                        ", ".join(
                            part
                            for part in (
                                _metric_summary(k, v) for k, v in (e.get("metrics") or {}).items()
                            )
                            if part
                        ),
                    ]
                    for e in (ranked or leaderboard)
                ],
            ),
        )
        unranked = [e for e in leaderboard if e.get("rank") is None]
        if unranked:
            lines.append(
                "_Not ranked: "
                + "; ".join(
                    f"{e.get('label') or e.get('model')} ({e.get('error') or 'no comparable metric'})"
                    for e in unranked
                )
                + "._"
            )
            lines.append("")

        if self.fairness:
            lines += _section("Fairness Report", _table(["Key", "Value"], [[k, _redact(v)] for k, v in self.fairness.items()]))

        lines += _checks_section(self.checks)

        if self.dataset_card:
            lines += _section(
                "Training data",
                [
                    (
                        f"This Model was trained on `{self.dataset_card.get('version_id')}`, "
                        f"Dataset Card v{self.dataset_card.get('number')} "
                        f"({self.dataset_card.get('row_count'):,} rows, "
                        f"origin {self.dataset_card.get('origin')})."
                    ),
                    "",
                    "_See its Dataset Card for lineage, Provenance, Jev Questions, review and Checks._",
                    "",
                ],
            )

        lines += _section(
            "Known limitations",
            _bullets(
                [
                    REPRODUCIBILITY_CAVEAT,
                    "The leaderboard ranks Models on a single held-out split; the differences between the top few are within the noise of that split size.",
                    "A Model fitted on synthetic rows learns whatever the Generator and Jev produced, including their biases. A high score is not evidence the task is solved.",
                    "This Card records what was fitted, not the estimator itself.",
                ]
            ),
        )
        return "\n".join(lines).rstrip() + "\n"

    def to_dict(self) -> dict[str, Any]:
        return _redact(
            {
                "kind": "model_card",
                "run_id": self.run_id,
                "version_id": self.run.get("version_id"),
                "target": (self.run.get("setup") or {}).get("target"),
                "task_type": (self.run.get("setup") or {}).get("task_type"),
                "seed": self.run.get("seed"),
                "primary_metric": self.run.get("primary_metric"),
                "leaderboard": self.run.get("leaderboard"),
                "fairness": self.fairness,
                "dataset_card": self.dataset_card,
                "checks": self.checks,
                "reproducibility": REPRODUCIBILITY_CAVEAT,
                "library_versions": _library_versions(),
            }
        )


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metric_summary(name: str, value: Any) -> str:
    """One metric as `name value`, or just `name` when it could not be measured.

    Metrics arrive as `{value, reason}` but a leaderboard may also carry scalar
    extras (counts, support sizes) alongside them, so the shape is checked rather
    than assumed.
    """
    if name == "confusion_matrix":
        return ""  # rendered as its own table, not inlined
    if isinstance(value, Mapping):
        return f"{name} {_fmt(value.get('value'))}"
    return f"{name} {_fmt(value)}"


# -- building a Card from the store -------------------------------------------


def dataset_card_for(
    store: Any,
    version_id: str,
    *,
    checks: list[dict[str, Any]] | None = None,
    review: dict[str, Any] | None = None,
    pii: dict[str, Any] | None = None,
    fairness: dict[str, Any] | None = None,
) -> DatasetCard:
    version = store.get_version(version_id)
    parent = None
    if version.parent_id:
        try:
            parent_version = store.get_version(version.parent_id)
            parent = {
                "id": parent_version.id,
                "number": parent_version.number,
                "origin": parent_version.origin,
            }
        except Exception:  # noqa: BLE001 - a pruned parent must not break the Card
            parent = {"id": version.parent_id}
    review_stats = dict(review or {})
    if not review_stats:
        stored = (version.meta or {}).get("review") or {}
        review_stats = {
            "accepted": (stored.get("outcome") or {}).get("accepted"),
            "overridden": (stored.get("outcome") or {}).get("overridden"),
            "excluded_rows": (stored.get("outcome") or {}).get("excluded_rows"),
            "still unreviewed": stored.get("unreviewed_count"),
        }
    return DatasetCard(
        version_id=version.id,
        project_id=version.project_id,
        number=version.number,
        origin=version.origin,
        row_count=version.row_count,
        columns=[c.to_dict() if hasattr(c, "to_dict") else dict(c) for c in version.columns],
        provenance_summary=dict(version.provenance_summary or {}),
        seed=version.seed,
        created_at=version.created_at,
        meta=dict(version.meta or {}),
        checks=list(checks or []),
        review=review_stats,
        pii=dict(pii or {}),
        parent=parent,
        fairness=fairness,
    )


def model_card_for(
    run: dict[str, Any],
    *,
    checks: list[dict[str, Any]] | None = None,
    fairness: dict[str, Any] | None = None,
    dataset_card: dict[str, Any] | None = None,
) -> ModelCard:
    return ModelCard(
        run=run,
        checks=list(checks or []),
        fairness=fairness,
        dataset_card=dataset_card,
    )


__all__ = [
    "REDACTED",
    "REPRODUCIBILITY_CAVEAT",
    "DatasetCard",
    "ModelCard",
    "dataset_card_for",
    "model_card_for",
]
