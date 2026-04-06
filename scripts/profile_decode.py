from __future__ import annotations

import asyncio

import torch

from engine.config import EngineConfig
from engine.runtime import HelixEngine


async def _run() -> None:
    config = EngineConfig(
        use_toy_backend=True,
        device="cuda",
        enable_cuda_graph_decode=True,
        max_decode_batch_size=8,
        prefill_chunk_size=128,
        kv_total_blocks=2048,
    )
    engine = HelixEngine(config)
    await engine.start()

    prompts = [
        "Describe paged KV cache in two lines.",
        "Why does continuous batching improve throughput?",
        "Explain chunked prefill and decode priority.",
        "How can CUDA Graph reduce decode overhead?",
    ]

    for prompt in prompts:
        await engine.generate(prompt, max_new_tokens=32, temperature=0.0)

    if torch.cuda.is_available():
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("decode_profile")

    await asyncio.gather(
        *(engine.generate(prompt, max_new_tokens=64, temperature=0.0) for prompt in prompts * 8)
    )

    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()

    await engine.stop()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
