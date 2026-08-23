from __future__ import annotations

import asyncio
import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field

from tickyantra.config import Settings


class QueueTimeoutError(RuntimeError):
    pass


@dataclass
class _Waiter:
    sequence: int
    prefix_key: str
    queued_at: float
    event: asyncio.Event = field(default_factory=asyncio.Event)
    granted: bool = False


@dataclass(frozen=True)
class Admission:
    prefix_key: str
    queue_wait_s: float
    admitted_at: float


class SLOAdmissionController:
    """Bounded, prefix-aware admission with additive-increase/decrease control."""

    def __init__(self, settings: Settings) -> None:
        settings.validate()
        self.settings = settings
        self._active = 0
        self._sequence = 0
        self._waiters: list[_Waiter] = []
        self._lock = asyncio.Lock()
        self._limit = self._initial_limit()
        self._ttft_ms: deque[float] = deque(maxlen=settings.sample_window)
        self._hot_prefixes: Counter[str] = Counter()
        self._hot_events: deque[tuple[float, str]] = deque()
        self._admitted = 0
        self._rejected = 0
        self._adjustments = 0

    def _initial_limit(self) -> int:
        if self.settings.mode == "native":
            return 1_000_000
        if self.settings.mode == "static":
            return self.settings.static_limit
        return self.settings.adaptive_initial_limit

    async def acquire(self, prefix_key: str) -> Admission:
        started = time.perf_counter()
        if self.settings.mode == "native":
            async with self._lock:
                self._active += 1
                self._admitted += 1
            return Admission(prefix_key, 0.0, time.perf_counter())

        async with self._lock:
            if self._active < self._limit and not self._waiters:
                self._active += 1
                self._admitted += 1
                return Admission(prefix_key, 0.0, time.perf_counter())
            self._sequence += 1
            waiter = _Waiter(self._sequence, prefix_key, time.perf_counter())
            self._waiters.append(waiter)

        try:
            await asyncio.wait_for(
                waiter.event.wait(), timeout=self.settings.queue_timeout_ms / 1000.0
            )
        except TimeoutError as exc:
            async with self._lock:
                if not waiter.granted:
                    self._waiters = [item for item in self._waiters if item is not waiter]
                    self._rejected += 1
                    raise QueueTimeoutError("admission deadline exceeded") from exc
            # The grant won the race with timeout; continue as admitted.

        return Admission(prefix_key, time.perf_counter() - started, time.perf_counter())

    async def release(self, admission: Admission, *, ttft_ms: float | None) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)
            if ttft_ms is not None and math.isfinite(ttft_ms):
                self._ttft_ms.append(ttft_ms)
            self._record_hot_prefix(admission.prefix_key)
            self._maybe_adjust_limit()
            self._grant_waiters()

    def _record_hot_prefix(self, prefix_key: str) -> None:
        now = time.monotonic()
        self._hot_events.append((now, prefix_key))
        self._hot_prefixes[prefix_key] += 1
        cutoff = now - self.settings.hot_prefix_ttl_s
        while self._hot_events and self._hot_events[0][0] < cutoff:
            _, expired = self._hot_events.popleft()
            self._hot_prefixes[expired] -= 1
            if self._hot_prefixes[expired] <= 0:
                del self._hot_prefixes[expired]

    def _maybe_adjust_limit(self) -> None:
        if self.settings.mode != "adaptive" or len(self._ttft_ms) < self.settings.sample_window:
            return
        ordered = sorted(self._ttft_ms)
        p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
        target = float(self.settings.target_ttft_ms)
        old = self._limit
        if p95 > target * 1.10:
            self._limit = max(self.settings.adaptive_min_limit, self._limit - 1)
        elif p95 < target * 0.75 and self._waiters:
            self._limit = min(self.settings.adaptive_max_limit, self._limit + 1)
        if self._limit != old:
            self._adjustments += 1
        self._ttft_ms.clear()

    def _grant_waiters(self) -> None:
        while self._waiters and self._active < self._limit:
            now = time.perf_counter()

            def score(waiter: _Waiter, scored_at: float = now) -> tuple[int, float, int]:
                age = scored_at - waiter.queued_at
                # Requests waiting over 25% of their deadline bypass affinity.
                fairness_due = age >= self.settings.queue_timeout_ms / 4000.0
                hot = self._hot_prefixes.get(waiter.prefix_key, 0) > 0
                return (0 if fairness_due else 1, 0 if hot else 1, waiter.sequence)

            waiter = min(self._waiters, key=score)
            self._waiters.remove(waiter)
            waiter.granted = True
            self._active += 1
            self._admitted += 1
            waiter.event.set()

    async def snapshot(self) -> dict[str, int | float | str]:
        async with self._lock:
            return {
                "mode": self.settings.mode,
                "active": self._active,
                "limit": self._limit,
                "queued": len(self._waiters),
                "admitted_total": self._admitted,
                "rejected_total": self._rejected,
                "limit_adjustments": self._adjustments,
                "hot_prefixes": len(self._hot_prefixes),
                "target_ttft_ms": self.settings.target_ttft_ms,
            }
