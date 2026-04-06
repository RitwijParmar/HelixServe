from __future__ import annotations

import asyncio
import copy
import math
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from cache.allocator import KVBlockAllocator, OutOfKVBlocksError
from cache.prefix_cache import PrefixCache, PrefixCacheEntry
from engine.config import EngineConfig
from engine.request import InferenceRequest, RequestHandle, RequestState, StreamEvent
from metrics.registry import EngineMetrics
from model.backend import DecoderBackend


class ContinuousBatchScheduler:
    def __init__(
        self,
        *,
        config: EngineConfig,
        backend: DecoderBackend,
        allocator: KVBlockAllocator,
        prefix_cache: PrefixCache,
        metrics: EngineMetrics,
    ) -> None:
        self.config = config
        self.backend = backend
        self.allocator = allocator
        self.prefix_cache = prefix_cache
        self.metrics = metrics

        self._incoming: asyncio.Queue[Tuple[InferenceRequest, RequestHandle]] = asyncio.Queue()
        self._requests: Dict[str, InferenceRequest] = {}
        self._handles: Dict[str, RequestHandle] = {}

        self._prefill_queue: Deque[str] = deque()
        self._decode_queue: Deque[str] = deque()

        self._running = False
        self._stop_event = asyncio.Event()

    def inflight_count(self) -> int:
        return len(self._requests) + self._incoming.qsize()

    def queue_depth(self) -> int:
        return self._incoming.qsize() + len(self._prefill_queue) + len(self._decode_queue)

    def _should_reject_for_backpressure(self) -> bool:
        if self.inflight_count() >= self.config.max_queue_size:
            return True
        stats = self.allocator.stats()
        if stats.get("memory_pressure", 0.0) > 0.97:
            return True
        return False

    async def submit(self, request: InferenceRequest, handle: RequestHandle) -> None:
        if self._should_reject_for_backpressure():
            request.state = RequestState.REJECTED
            await handle.push(
                StreamEvent(
                    event="rejected",
                    request_id=request.request_id,
                    payload={"reason": "backpressure"},
                )
            )
            return

        await self._incoming.put((request, handle))

    async def run_forever(self) -> None:
        self._running = True
        self._stop_event.clear()

        while not self._stop_event.is_set():
            await self._admit_incoming()
            await self._decode_step()
            await self._prefill_step()
            self._refresh_metrics()
            await asyncio.sleep(self.config.scheduler_sleep_s)

        self._running = False

    async def stop(self) -> None:
        self._stop_event.set()

    async def _admit_incoming(self) -> None:
        while True:
            try:
                request, handle = self._incoming.get_nowait()
            except asyncio.QueueEmpty:
                break

            if request.request_id in self._requests:
                await handle.push(
                    StreamEvent(
                        event="error",
                        request_id=request.request_id,
                        payload={"error": "duplicate request id"},
                    )
                )
                continue

            if self._should_reject_for_backpressure():
                request.state = RequestState.REJECTED
                await handle.push(
                    StreamEvent(
                        event="rejected",
                        request_id=request.request_id,
                        payload={"reason": "backpressure"},
                    )
                )
                self.metrics.requests_total.labels(status="rejected").inc()
                continue

            request.state = RequestState.PREFILL
            request.queue_entered_at = time.time()

            self._requests[request.request_id] = request
            self._handles[request.request_id] = handle
            self.metrics.register_request(request.request_id, request.created_at)

            prefix_entry = self.prefix_cache.lookup(request.prompt_tokens)
            if prefix_entry is not None:
                await self._apply_prefix_hit(request, prefix_entry)

            if request.prefill_cursor >= len(request.prompt_tokens):
                self._mark_prefill_complete(request)
            else:
                self._prefill_queue.append(request.request_id)

    async def _apply_prefix_hit(
        self,
        request: InferenceRequest,
        entry: PrefixCacheEntry,
    ) -> None:
        self.allocator.attach_blocks(request.request_id, entry.block_ids)
        request.prefill_cursor = min(entry.token_count, len(request.prompt_tokens))
        request.cached_prefix_tokens = request.prefill_cursor

        imported = False
        if entry.backend_state is not None:
            imported = self.backend.import_state(request.request_id, copy.deepcopy(entry.backend_state))

        if not imported and request.prefill_cursor > 0:
            self.backend.prefill(request.request_id, request.prompt_tokens[: request.prefill_cursor])

    def _mark_prefill_complete(self, request: InferenceRequest) -> None:
        request.state = RequestState.DECODE
        request.decode_started_at = time.time()
        request.force_new_decode_block = True
        if request.prompt_tokens:
            request.last_token = request.prompt_tokens[-1]
        else:
            request.last_token = self.backend.eos_token_id

        blocks = self.allocator.get_request_blocks(request.request_id)
        prompt_block_count = math.ceil(len(request.prompt_tokens) / self.config.kv_block_size)
        prompt_blocks = blocks[:prompt_block_count]

        if request.prompt_tokens and prompt_blocks:
            backend_state = self.backend.export_state(request.request_id)
            self.prefix_cache.insert(
                request.prompt_tokens,
                prompt_blocks,
                backend_state=backend_state,
            )

        self._decode_queue.append(request.request_id)

    async def _prefill_step(self) -> None:
        if not self._prefill_queue:
            return

        budget = max(self.config.max_num_batched_tokens, self.config.prefill_chunk_size)
        rotate = len(self._prefill_queue)

        while rotate > 0 and budget > 0 and self._prefill_queue:
            request_id = self._prefill_queue.popleft()
            rotate -= 1
            request = self._requests.get(request_id)
            if request is None or request.state != RequestState.PREFILL:
                continue

            remaining = len(request.prompt_tokens) - request.prefill_cursor
            if remaining <= 0:
                self._mark_prefill_complete(request)
                continue

            chunk = min(self.config.prefill_chunk_size, remaining, budget)
            if chunk <= 0:
                self._prefill_queue.append(request_id)
                continue

            chunk_tokens = request.prompt_tokens[request.prefill_cursor : request.prefill_cursor + chunk]

            try:
                if request.prefill_cursor == 0 and request.cached_prefix_tokens == 0:
                    self.metrics.mark_queue_delay(request.request_id)

                self.backend.prefill(request.request_id, chunk_tokens)
                self.allocator.append_tokens(request.request_id, chunk)
            except OutOfKVBlocksError:
                await self._reject_or_fail(
                    request,
                    status="rejected",
                    reason="kv-cache exhausted",
                )
                continue
            except Exception as exc:
                await self._reject_or_fail(
                    request,
                    status="error",
                    reason=f"prefill error: {exc}",
                )
                continue

            request.prefill_cursor += chunk
            budget -= chunk
            self.metrics.observe_prefill(chunk)

            if request.prefill_cursor >= len(request.prompt_tokens):
                self._mark_prefill_complete(request)
            else:
                self._prefill_queue.append(request.request_id)

    def _select_decode_group(self) -> List[InferenceRequest]:
        if not self._decode_queue:
            return []

        selected: List[InferenceRequest] = []
        deferred: List[str] = []

        # Take the first active decode request as the parameter anchor.
        anchor_req: Optional[InferenceRequest] = None
        while self._decode_queue and anchor_req is None:
            rid = self._decode_queue.popleft()
            req = self._requests.get(rid)
            if req is None or req.state != RequestState.DECODE:
                continue
            anchor_req = req

        if anchor_req is None:
            return []

        selected.append(anchor_req)
        max_batch = self.config.max_decode_batch_size

        while self._decode_queue and len(selected) < max_batch:
            rid = self._decode_queue.popleft()
            req = self._requests.get(rid)
            if req is None or req.state != RequestState.DECODE:
                continue

            same_policy = (
                req.params.temperature == anchor_req.params.temperature
                and req.params.top_k == anchor_req.params.top_k
            )
            if same_policy:
                selected.append(req)
            else:
                deferred.append(rid)

        for rid in reversed(deferred):
            self._decode_queue.appendleft(rid)

        return selected

    async def _decode_step(self) -> None:
        selected = self._select_decode_group()
        if not selected:
            self.metrics.set_decode_batch_size(0)
            return

        self.metrics.set_decode_batch_size(len(selected))

        request_ids = [req.request_id for req in selected]
        prev_tokens = [
            req.generated_tokens[-1]
            if req.generated_tokens
            else (req.last_token if req.last_token is not None else self.backend.eos_token_id)
            for req in selected
        ]

        temperature = selected[0].params.temperature
        top_k = selected[0].params.top_k

        try:
            next_tokens = self.backend.decode_batch(
                request_ids,
                prev_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        except Exception as exc:
            for req in selected:
                await self._reject_or_fail(req, status="error", reason=f"decode error: {exc}")
            return

        for req, token in zip(selected, next_tokens):
            try:
                self.allocator.append_tokens(
                    req.request_id,
                    1,
                    force_new_block=req.force_new_decode_block,
                )
                req.force_new_decode_block = False
            except OutOfKVBlocksError:
                await self._reject_or_fail(req, status="rejected", reason="kv-cache exhausted")
                continue

            token = int(token)
            req.generated_tokens.append(token)
            req.last_token = token

            text = self.backend.detokenize([token])
            handle = self._handles.get(req.request_id)
            if handle is not None:
                await handle.push(
                    StreamEvent(
                        event="token",
                        request_id=req.request_id,
                        payload={"token_id": token, "text": text},
                    )
                )

            if req.first_token_at is None:
                req.first_token_at = time.time()
            self.metrics.mark_token(req.request_id)

            if self._is_finished(req, token):
                await self._finish_request(req, status="ok")
            else:
                self._decode_queue.append(req.request_id)

    def _is_finished(self, request: InferenceRequest, token: int) -> bool:
        if token == self.backend.eos_token_id:
            return True
        if request.params.stop_token_ids and token in request.params.stop_token_ids:
            return True
        if len(request.generated_tokens) >= request.params.max_new_tokens:
            return True
        return False

    async def _reject_or_fail(
        self,
        request: InferenceRequest,
        *,
        status: str,
        reason: str,
    ) -> None:
        request.state = RequestState.ERROR if status == "error" else RequestState.REJECTED
        handle = self._handles.get(request.request_id)
        if handle is not None:
            await handle.push(
                StreamEvent(
                    event="error" if status == "error" else "rejected",
                    request_id=request.request_id,
                    payload={"reason": reason},
                )
            )
        await self._cleanup_request(request, status=status)

    async def _finish_request(self, request: InferenceRequest, *, status: str) -> None:
        request.state = RequestState.FINISHED
        request.finished_at = time.time()
        handle = self._handles.get(request.request_id)
        if handle is not None:
            await handle.push(
                StreamEvent(
                    event="done",
                    request_id=request.request_id,
                    payload={
                        "generated_tokens": len(request.generated_tokens),
                        "cached_prefix_tokens": request.cached_prefix_tokens,
                    },
                )
            )
        await self._cleanup_request(request, status=status)

    async def _cleanup_request(self, request: InferenceRequest, *, status: str) -> None:
        rid = request.request_id
        self.allocator.release_request(rid)
        self.backend.remove_request(rid)

        self._requests.pop(rid, None)
        self._handles.pop(rid, None)

        self.metrics.finish_request(rid, status=status)

    def _refresh_metrics(self) -> None:
        self.metrics.set_queue_depth(self.queue_depth())
        self.metrics.set_kv_stats(self.allocator.stats())
        self.metrics.set_prefix_stats(self.prefix_cache.stats())
