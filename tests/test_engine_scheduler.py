import asyncio

import pytest

from engine.config import EngineConfig
from engine.runtime import HelixEngine


@pytest.mark.asyncio
async def test_engine_generates_text() -> None:
    engine = HelixEngine(
        EngineConfig(
            use_toy_backend=True,
            device="cpu",
            kv_block_size=8,
            kv_total_blocks=256,
            max_decode_batch_size=4,
            prefill_chunk_size=16,
            max_num_batched_tokens=64,
            enable_cuda_graph_decode=False,
        )
    )

    await engine.start()
    try:
        text = await engine.generate("hello", max_new_tokens=8, temperature=0.0)
        assert isinstance(text, str)
        assert len(text) >= 0
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_prefix_cache_hit_emits_cached_tokens() -> None:
    engine = HelixEngine(
        EngineConfig(
            use_toy_backend=True,
            device="cpu",
            kv_block_size=4,
            kv_total_blocks=256,
            prefill_chunk_size=8,
            max_num_batched_tokens=32,
            prefix_cache_min_tokens=2,
            enable_cuda_graph_decode=False,
        )
    )

    await engine.start()
    try:
        prompt = "shared prefix prompt for cache test"

        first = await engine.submit(prompt=prompt, max_new_tokens=4)
        async for _ in first.stream():
            pass

        second = await engine.submit(prompt=prompt, max_new_tokens=4)
        done_payload = None
        async for event in second.stream():
            if event.event == "done":
                done_payload = event.payload

        assert done_payload is not None
        assert int(done_payload["cached_prefix_tokens"]) > 0
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_continuous_batching_handles_concurrency() -> None:
    engine = HelixEngine(
        EngineConfig(
            use_toy_backend=True,
            device="cpu",
            kv_block_size=8,
            kv_total_blocks=512,
            max_decode_batch_size=8,
            prefill_chunk_size=12,
            max_num_batched_tokens=96,
            enable_cuda_graph_decode=False,
        )
    )

    await engine.start()
    try:
        handles = [
            await engine.submit(prompt=f"request {i}", max_new_tokens=12, temperature=0.0)
            for i in range(12)
        ]

        await asyncio.gather(*(h.wait() for h in handles))
        stats = engine.stats()
        assert stats["scheduler"]["inflight"] == 0
    finally:
        await engine.stop()
