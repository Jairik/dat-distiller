"""The held-out test split.

One split per Training Run, recorded with its seed, so the leaderboard (#32) and
the Model Bundle (#34) can reproduce the same train/test rows later. For
classification the split is stratified: every class keeps rows on both sides.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_TEST_SIZE = 0.2


@dataclass
class Split:
    """Positional row indices for the two sides, plus how they were made."""

    train: np.ndarray
    test: np.ndarray
    seed: int
    test_size: float
    stratified: bool = False
    #: Row counts kept alongside the indices so the persisted plan round-trips
    #: even where the indices themselves are not stored.
    train_rows: int | None = None
    test_rows: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_rows": int(len(self.train) if self.train_rows is None else self.train_rows),
            "test_rows": int(len(self.test) if self.test_rows is None else self.test_rows),
            "seed": int(self.seed),
            "test_size": float(self.test_size),
            "stratified": bool(self.stratified),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Split:
        return cls(
            train=np.array([], dtype=int),
            test=np.array([], dtype=int),
            seed=int(payload.get("seed", 0)),
            test_size=float(payload.get("test_size", DEFAULT_TEST_SIZE)),
            stratified=bool(payload.get("stratified", False)),
            train_rows=int(payload.get("train_rows", 0)),
            test_rows=int(payload.get("test_rows", 0)),
        )


def test_row_count(rows: int, test_size: float) -> int:
    """How many rows the test split takes, always leaving train non-empty."""
    if rows < 2:
        return 0
    wanted = int(round(rows * test_size))
    return max(1, min(wanted, rows - 1))


def _key(value: Any) -> str:
    """A hashable, sortable label for one Target value."""
    if value is None or value is pd.NA:
        return "missing"
    if isinstance(value, (bool, np.bool_)):
        return f"bool:{bool(value)}"
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}"
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "missing"
        return f"num:{float(value):.6g}"
    return f"str:{value}"


def _class_groups(values: Sequence[Any]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, value in enumerate(values):
        groups.setdefault(_key(value), []).append(index)
    return groups


def _stratified_sides(
    groups: dict[str, list[int]], test_size: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Per class: shuffle, take a proportional share for the test split.

    A class of two or more rows always puts at least one row in the test split
    and keeps one for training, so a rare class is measured rather than lost —
    which is why the test share can come out slightly above ``test_size`` on a
    frame with very small classes.
    """
    counts: dict[str, int] = {}
    for key, positions in groups.items():
        count = len(positions)
        take = int(round(count * test_size))
        if count > 1:
            take = max(1, min(take, count - 1))
        else:
            take = 0
        counts[key] = take
    train: list[int] = []
    test: list[int] = []
    for key, positions in groups.items():
        shuffled = list(positions)
        rng.shuffle(shuffled)
        take = counts[key]
        test.extend(shuffled[:take])
        train.extend(shuffled[take:])
    return np.array(sorted(train), dtype=int), np.array(sorted(test), dtype=int)


def split_indices(
    rows: int,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = 0,
    stratify: Sequence[Any] | None = None,
) -> Split:
    """Split ``rows`` positions into train/test, reproducibly from ``seed``.

    ``stratify`` is the Target's values for a classification Training Run; pass
    ``None`` for regression. Frames too small to hold out a row get everything
    in train rather than raising — the estimate endpoint reports it.
    """
    if rows < 0:
        raise ValueError("row count must not be negative")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    rng = np.random.default_rng(seed)
    if rows < 2:
        return Split(
            train=np.arange(rows, dtype=int),
            test=np.array([], dtype=int),
            seed=seed,
            test_size=test_size,
            stratified=False,
        )
    if stratify is not None and len(_class_groups(stratify)) > 1:
        train, test = _stratified_sides(_class_groups(stratify), test_size, rng)
        return Split(train=train, test=test, seed=seed, test_size=test_size, stratified=True)
    order = rng.permutation(rows)
    holdout = test_row_count(rows, test_size)
    test = np.sort(order[:holdout])
    train = np.sort(order[holdout:])
    return Split(train=train, test=test, seed=seed, test_size=test_size, stratified=False)
