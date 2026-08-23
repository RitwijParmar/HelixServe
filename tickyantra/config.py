from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    upstream_url: str = "http://127.0.0.1:30000"
    mode: str = "adaptive"
    static_limit: int = 16
    adaptive_min_limit: int = 2
    adaptive_max_limit: int = 32
    adaptive_initial_limit: int = 8
    target_ttft_ms: int = 500
    queue_timeout_ms: int = 2_000
    sample_window: int = 32
    prefix_chars: int = 128
    hot_prefix_ttl_s: int = 60
    api_key: str = ""

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            upstream_url=os.getenv("TICKYANTRA_UPSTREAM_URL", "http://127.0.0.1:30000").rstrip("/"),
            mode=os.getenv("TICKYANTRA_MODE", "adaptive").lower(),
            static_limit=_env_int("TICKYANTRA_STATIC_LIMIT", 16),
            adaptive_min_limit=_env_int("TICKYANTRA_ADAPTIVE_MIN", 2),
            adaptive_max_limit=_env_int("TICKYANTRA_ADAPTIVE_MAX", 32),
            adaptive_initial_limit=_env_int("TICKYANTRA_ADAPTIVE_INITIAL", 8),
            target_ttft_ms=_env_int("TICKYANTRA_TARGET_TTFT_MS", 500),
            queue_timeout_ms=_env_int("TICKYANTRA_QUEUE_TIMEOUT_MS", 2_000),
            sample_window=_env_int("TICKYANTRA_SAMPLE_WINDOW", 32),
            prefix_chars=_env_int("TICKYANTRA_PREFIX_CHARS", 128),
            hot_prefix_ttl_s=_env_int("TICKYANTRA_HOT_PREFIX_TTL_S", 60),
            api_key=os.getenv("TICKYANTRA_API_KEY", ""),
        )

    def validate(self) -> None:
        if self.mode not in {"native", "static", "adaptive"}:
            raise ValueError("TICKYANTRA_MODE must be native, static, or adaptive")
        if min(self.static_limit, self.adaptive_min_limit, self.adaptive_initial_limit) < 1:
            raise ValueError("concurrency limits must be positive")
        if not self.adaptive_min_limit <= self.adaptive_initial_limit <= self.adaptive_max_limit:
            raise ValueError("adaptive limits must satisfy min <= initial <= max")
        if self.target_ttft_ms <= 0 or self.queue_timeout_ms <= 0:
            raise ValueError("SLO and queue timeout must be positive")
