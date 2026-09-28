"""Concurrency pool with retries and exponential backoff.

Used by Labeling (many small calls to Jev) and anywhere else that must
process N items through a worker function concurrently, retry failures with
backoff, report per-item outcomes, and stop promptly on cancellation.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, TypeVar

T = TypeVar("T")

ShouldCancel = Callable[[], bool] | None


@dataclass
class ItemResult:
    index: int
    item: Any
    value: Any = None
    error: str | None = None
    attempts: int = 1
    skipped: bool = False  # not attempted because of cancellation


@dataclass
class PoolOutcome:
    results: list[ItemResult] = field(default_factory=list)
    cancelled: bool = False

    @property
    def failures(self) -> list[ItemResult]:
        return [r for r in self.results if r.error is not None]

    @property
    def values(self) -> list[Any]:
        return [r.value for r in self.results if r.error is None and not r.skipped]


def run_pool(
    items: Sequence[T],
    worker: Callable[[T], Any],
    *,
    concurrency: int = 4,
    retries: int = 2,
    backoff_base: float = 0.2,
    should_cancel: ShouldCancel = None,
    on_item_done: Callable[[ItemResult], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> PoolOutcome:
    """Run ``worker`` over ``items`` in a thread pool.

    Each item is retried up to ``retries`` times with exponential backoff
    (``backoff_base * 2**attempt`` seconds). A true from ``should_cancel``
    stops scheduling new items (they come back ``skipped``); already running
    items finish. Never raises for worker errors — inspect the outcome.
    """
    outcome = PoolOutcome()
    cancelled = False

    def run_one(index: int, item: T) -> ItemResult:
        nonlocal cancelled
        result = ItemResult(index=index, item=item)
        for attempt in range(retries + 1):
            if should_cancel is not None and should_cancel():
                cancelled = True
                result.skipped = True
                return result
            result.attempts = attempt + 1
            try:
                result.value = worker(item)
                result.error = None
                return result
            except Exception as exc:  # noqa: BLE001 - report, never propagate
                result.error = f"{type(exc).__name__}: {exc}"
                if attempt < retries:
                    sleep(backoff_base * (2**attempt))
        return result

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures: list[Future[ItemResult]] = []
        for index, item in enumerate(items):
            if should_cancel is not None and should_cancel():
                cancelled = True
                break
            futures.append(executor.submit(run_one, index, item))
        for future in futures:
            result = future.result()
            outcome.results.append(result)
            if on_item_done is not None:
                on_item_done(result)

    if should_cancel is not None and should_cancel():
        cancelled = True
    outcome.cancelled = cancelled
    outcome.results.sort(key=lambda r: r.index)
    return outcome
