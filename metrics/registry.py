from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


@dataclass
class RequestTiming:
    created_at: float
    first_token_at: Optional[float] = None


class EngineMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()

        self.requests_total = Counter(
            "helixserve_requests_total",
            "Total requests",
            ["status"],
            registry=self.registry,
        )
        self.tokens_generated_total = Counter(
            "helixserve_tokens_generated_total",
            "Total generated tokens",
            registry=self.registry,
        )
        self.prefill_tokens_total = Counter(
            "helixserve_prefill_tokens_total",
            "Total prefill tokens processed",
            registry=self.registry,
        )
        self.prefill_chunks_total = Counter(
            "helixserve_prefill_chunks_total",
            "Total prefill chunks",
            registry=self.registry,
        )

        self.ttft_seconds = Histogram(
            "helixserve_ttft_seconds",
            "Time to first token",
            buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
            registry=self.registry,
        )
        self.itl_seconds = Histogram(
            "helixserve_itl_seconds",
            "Inter-token latency",
            buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
            registry=self.registry,
        )
        self.request_latency_seconds = Histogram(
            "helixserve_request_latency_seconds",
            "End-to-end request latency",
            buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20),
            registry=self.registry,
        )
        self.queue_delay_seconds = Histogram(
            "helixserve_queue_delay_seconds",
            "Queue delay before first prefill step",
            buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1),
            registry=self.registry,
        )

        self.active_requests = Gauge(
            "helixserve_active_requests",
            "Current active requests",
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            "helixserve_queue_depth",
            "Admission queue depth",
            registry=self.registry,
        )
        self.active_decode_batch_size = Gauge(
            "helixserve_active_decode_batch_size",
            "Current decode batch size",
            registry=self.registry,
        )
        self.kv_used_tokens = Gauge(
            "helixserve_kv_used_tokens",
            "Used tokens in live KV blocks",
            registry=self.registry,
        )
        self.kv_utilization = Gauge(
            "helixserve_kv_block_utilization",
            "KV block utilization ratio",
            registry=self.registry,
        )
        self.kv_fragmentation_tokens = Gauge(
            "helixserve_kv_internal_fragmentation_tokens",
            "Internal fragmentation in token slots",
            registry=self.registry,
        )
        self.prefix_cache_entries = Gauge(
            "helixserve_prefix_cache_entries",
            "Prefix cache entries",
            registry=self.registry,
        )
        self.prefix_cache_hit_rate = Gauge(
            "helixserve_prefix_cache_hit_rate",
            "Prefix cache hit rate",
            registry=self.registry,
        )

        self._request_timings: Dict[str, RequestTiming] = {}
        self._request_last_token_ts: Dict[str, float] = {}

    def register_request(self, request_id: str, created_at: float) -> None:
        self._request_timings[request_id] = RequestTiming(created_at=created_at)
        self.active_requests.inc()

    def mark_queue_delay(self, request_id: str) -> None:
        timing = self._request_timings.get(request_id)
        if timing is None:
            return
        delay = max(time.time() - timing.created_at, 0.0)
        self.queue_delay_seconds.observe(delay)

    def mark_token(self, request_id: str) -> None:
        now = time.time()
        self.tokens_generated_total.inc()
        timing = self._request_timings.get(request_id)
        if timing and timing.first_token_at is None:
            timing.first_token_at = now
            self.ttft_seconds.observe(max(now - timing.created_at, 0.0))

        last = self._request_last_token_ts.get(request_id)
        if last is not None:
            self.itl_seconds.observe(max(now - last, 0.0))
        self._request_last_token_ts[request_id] = now

    def finish_request(self, request_id: str, status: str) -> None:
        now = time.time()
        self.requests_total.labels(status=status).inc()
        timing = self._request_timings.pop(request_id, None)
        self._request_last_token_ts.pop(request_id, None)
        if timing is not None:
            self.request_latency_seconds.observe(max(now - timing.created_at, 0.0))
        self.active_requests.dec()

    def observe_prefill(self, tokens: int) -> None:
        self.prefill_tokens_total.inc(tokens)
        self.prefill_chunks_total.inc()

    def set_queue_depth(self, depth: int) -> None:
        self.queue_depth.set(depth)

    def set_decode_batch_size(self, batch_size: int) -> None:
        self.active_decode_batch_size.set(batch_size)

    def set_kv_stats(self, stats: Dict[str, float]) -> None:
        self.kv_used_tokens.set(float(stats.get("used_tokens", 0)))
        self.kv_fragmentation_tokens.set(float(stats.get("internal_waste_tokens", 0)))
        self.kv_utilization.set(float(stats.get("block_utilization", 0.0)))

    def set_prefix_stats(self, stats: Dict[str, float]) -> None:
        self.prefix_cache_entries.set(float(stats.get("entries", 0)))
        self.prefix_cache_hit_rate.set(float(stats.get("hit_rate", 0.0)))

    def render(self) -> bytes:
        return generate_latest(self.registry)
