from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "tickyantra_requests_total", "Requests by outcome", ["outcome"], registry=self.registry
        )
        self.queue_wait = Histogram(
            "tickyantra_queue_wait_seconds",
            "Admission queue wait",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
            registry=self.registry,
        )
        self.ttft = Histogram(
            "tickyantra_ttft_seconds",
            "Client-observed time to first token, including admission queueing",
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
            registry=self.registry,
        )
        self.e2e = Histogram(
            "tickyantra_e2e_seconds",
            "End-to-end gateway latency",
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
            registry=self.registry,
        )
        self.active = Gauge(
            "tickyantra_active_requests", "Active upstream requests", registry=self.registry
        )
        self.limit = Gauge(
            "tickyantra_concurrency_limit", "Current concurrency limit", registry=self.registry
        )
        self.queued = Gauge(
            "tickyantra_queued_requests", "Queued requests", registry=self.registry
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
