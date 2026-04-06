from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict
from typing import Any, Dict, Optional

from cache.allocator import KVBlockAllocator
from cache.prefix_cache import PrefixCache
from cuda_ext import has_compiled_cuda_ext
from engine.config import EngineConfig
from engine.request import GenerationParams, InferenceRequest, RequestHandle
from engine.scheduler import ContinuousBatchScheduler
from metrics.registry import EngineMetrics
from model.factory import build_backend


class HelixEngine:
    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()

        self.metrics = EngineMetrics()
        self.backend = build_backend(self.config)

        self.allocator = KVBlockAllocator(
            total_blocks=self.config.kv_total_blocks,
            block_size=self.config.kv_block_size,
        )

        self.prefix_cache = PrefixCache(
            block_size=self.config.kv_block_size,
            max_entries=self.config.prefix_cache_max_entries,
            max_cached_tokens=self.config.prefix_cache_max_tokens,
            min_prefix_tokens=self.config.prefix_cache_min_tokens,
            cacheable_prefix_lengths=self.config.prefix_cache_lengths,
            pin_blocks=self.allocator.pin_blocks,
            unpin_blocks=self.allocator.unpin_blocks,
        )

        self.scheduler = ContinuousBatchScheduler(
            config=self.config,
            backend=self.backend,
            allocator=self.allocator,
            prefix_cache=self.prefix_cache,
            metrics=self.metrics,
        )

        self._task: asyncio.Task | None = None
        self._start_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._start_lock:
            if self._task is not None and not self._task.done():
                return
            self._task = asyncio.create_task(self.scheduler.run_forever(), name="helix-scheduler")

    async def stop(self) -> None:
        await self.scheduler.stop()
        if self._task is not None:
            await self._task
            self._task = None

    async def submit(
        self,
        *,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: int = 0,
        stop_token_ids: Optional[list[int]] = None,
        request_id: Optional[str] = None,
    ) -> RequestHandle:
        await self.start()

        rid = request_id or str(uuid.uuid4())
        prompt_tokens = self.backend.tokenize(prompt)
        params = GenerationParams(
            max_new_tokens=max_new_tokens or self.config.default_max_new_tokens,
            temperature=(
                self.config.default_temperature if temperature is None else float(temperature)
            ),
            top_k=top_k,
            stop_token_ids=stop_token_ids or [],
        )
        request = InferenceRequest(
            request_id=rid,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            params=params,
        )
        handle = RequestHandle(request)
        await self.scheduler.submit(request, handle)
        return handle

    async def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: int = 0,
    ) -> str:
        handle = await self.submit(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        return await handle.collect_text()

    def metrics_payload(self) -> bytes:
        return self.metrics.render()

    def stats(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "kv": self.allocator.stats(),
            "prefix_cache": self.prefix_cache.stats(),
            "scheduler": {
                "inflight": self.scheduler.inflight_count(),
                "queue_depth": self.scheduler.queue_depth(),
            },
            "backend": {
                "type": type(self.backend).__name__,
                "device": self.backend.device,
                "cuda_graph_decode": self.backend.supports_cuda_graph_decode,
                "cuda_cpp_extension_loaded": has_compiled_cuda_ext(),
            },
        }
