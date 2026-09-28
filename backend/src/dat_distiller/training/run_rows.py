"""A stored Training Run's rows, rebuilt — and proved to be its own.

A Training Run does not train on the Dataset Version as stored. ``build_setup``
first drops the rows it will not use: rows with no Target value, and (past the
Review threshold) rows whose labelling a human has not accepted. The split it
records, and every index in it, addresses **that filtered frame** — not the
Dataset Version.

So anything that wants to act on a finished run's rows has to rebuild the
filtered frame, and has to check it rebuilt *the same* one. Reading the Dataset
Version directly is the tempting mistake, and it is silent: the indices are all
still in range, they just point at the wrong rows.

This module is the one place that does the rebuild, so the leaderboard, the
Fairness Report, the plots and the Model Bundle cannot disagree about which rows
a run used. Consumers that want a refusal in their own voice (a Fairness Report
says ``FairnessError``, an endpoint says HTTP 422) catch :class:`ValueError` and
re-raise; nobody re-implements the check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..store.provenance import PROVENANCE_COLUMN
from .preprocess import transform_with_spec
from .setup import DEFAULT_REVIEW_THRESHOLD, TrainingSetup, build_setup
from .train import TrainingRunRequest, split_request
from .trainers import (
    CLASSIFICATION,
    TrainingMatrix,
    assert_held_out,
    encode_classes,
)

__all__ = ["RunContext", "resolve_run"]


@dataclass
class RunContext:
    """The stored Training Run, rebuilt into the exact rows and matrix it used."""

    run: dict[str, Any]
    setup: TrainingSetup
    request: TrainingRunRequest
    #: The Dataset Version as stored, before the run's own row filtering. This is
    #: what group sizes and Dataset-wide counts are measured over — a Sensitive
    #: Attribute's groups are a property of the version, not of the kept rows.
    source_frame: pd.DataFrame
    #: The rows the run trained on and scored on, after that filtering. Every
    #: index in ``train_indices``/``test_indices`` addresses **this** frame.
    frame: pd.DataFrame
    preprocessing: dict[str, Any]
    train_indices: list[int]
    test_indices: list[int]
    #: The Target's classes, sorted — the codes a prediction is aligned to.
    classes: list[str] = field(default_factory=list)

    @property
    def data_frame(self) -> pd.DataFrame:
        """The kept rows, without the bookkeeping column.

        What the run trained on, minus the provenance blob. Everything that
        indexes rows or describes a Model's inputs wants this and not
        :attr:`frame`: a bundle's ``input_schema.json`` tells a caller what they
        have to supply, and nobody supplies a provenance blob.
        """
        return self.frame.drop(columns=[PROVENANCE_COLUMN], errors="ignore")

    @property
    def train_frame(self) -> pd.DataFrame:
        return self.frame.iloc[list(self.train_indices)]

    @property
    def test_frame(self) -> pd.DataFrame:
        return self.frame.iloc[list(self.test_indices)]

    def test_matrix(self) -> np.ndarray:
        """The held-out matrix, from the run's **persisted** preprocessing spec."""
        return transform_with_spec(self.test_frame, self.preprocessing)

    def train_matrix(self) -> TrainingMatrix:
        """The training matrix, built by the same closed constructor the run used."""
        matrix = TrainingMatrix.from_split(
            frame=self.frame, spec=self.preprocessing, setup=self.setup
        )
        assert_held_out(matrix.rows, self.test_indices)
        return matrix


def persisted_run(job: Any) -> dict[str, Any]:
    """The run dict from a finished Training Run, however it was stored."""
    checkpoint = job.checkpoint or {}
    run = checkpoint.get("run")
    if isinstance(run, dict) and run:
        return dict(run)
    if isinstance(job.result, dict) and job.result:
        return dict(job.result)
    raise ValueError(
        "this Training Run has no persisted leaderboard yet, so there is nothing to work from. "
        "Wait for the run to finish, or read its error."
    )


def resolve_run(store: Any, job: Any) -> RunContext:
    """Rebuild the run's rows, split and preprocessing — and prove they are its own.

    The split is re-derived from the run's own request with the run's own review
    threshold, then compared against the indices the run recorded. A mismatch
    means the run's stored split cannot be reproduced, and the caller is refused
    rather than allowed to work against different rows than the leaderboard used.

    Raises :class:`ValueError` with a message meant to be shown to a person.
    """
    run = persisted_run(job)
    params = job.params or {}
    if not params:
        raise ValueError(
            "this Training Run recorded no request, so its split cannot be reproduced; re-run "
            "the Training Run"
        )
    try:
        request = TrainingRunRequest.model_validate(params)
    except Exception as exc:
        raise ValueError(
            f"this Training Run's request can no longer be read ({exc}); re-run the Training Run"
        ) from exc

    version_id = str(run.get("version_id") or request.version_id)
    if not version_id:
        raise ValueError("this Training Run does not record its Dataset Version")
    source_frame = store.load_dataframe(version_id, include_provenance=True)
    stored_setup = run.get("setup") or {}
    # The run's own threshold, not today's: a changed setting must not move the
    # rows a report measures or a bundle refits.
    review_threshold = float(stored_setup.get("review_threshold", DEFAULT_REVIEW_THRESHOLD))
    try:
        resolved = build_setup(
            split_request(request), source_frame, review_threshold=review_threshold
        )
    except ValueError as exc:
        raise ValueError(
            f"this Training Run's setup can no longer be reproduced ({exc}); re-run the "
            "Training Run"
        ) from exc

    setup = resolved.setup
    stored_test = [int(i) for i in ((run.get("test_split") or {}).get("indices") or [])]
    stored_train = [int(i) for i in ((run.get("training_split") or {}).get("rows") or [])]
    if not stored_test:
        raise ValueError(
            "this Training Run recorded no held-out rows, so there is nothing to work from; "
            "raise test_size and re-run it"
        )
    if [int(i) for i in setup.split.test] != stored_test:
        raise ValueError(
            "this Training Run's held-out rows cannot be reproduced from its own request and "
            "seed, so this would not work on the same rows as its leaderboard; re-run the "
            "Training Run"
        )
    if stored_train and [int(i) for i in setup.split.train] != stored_train:
        raise ValueError(
            "this Training Run's training rows cannot be reproduced from its own request and "
            "seed, so the Model would be refitted on different data; re-run the Training Run"
        )

    preprocessing = run.get("preprocessing") or {}
    if not preprocessing:
        raise ValueError(
            "this Training Run recorded no preprocessing pipeline, so the rows it was fitted "
            "on cannot be rebuilt; re-run the Training Run"
        )

    train_indices = [int(i) for i in setup.split.train]
    classes: list[str] = []
    if setup.task_type == CLASSIFICATION:
        _codes, classes = encode_classes(
            resolved.frame.iloc[train_indices][setup.target].to_numpy()
        )
    return RunContext(
        run=run,
        setup=setup,
        request=request,
        source_frame=source_frame,
        frame=resolved.frame,
        preprocessing=preprocessing,
        train_indices=train_indices,
        test_indices=[int(i) for i in setup.split.test],
        classes=classes,
    )
