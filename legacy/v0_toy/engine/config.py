from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class EngineConfig:
    model_name: str = "sshleifer/tiny-gpt2"
    use_toy_backend: bool = True
    device: str = "cuda"

    # KV paging
    kv_block_size: int = 16
    kv_total_blocks: int = 4096

    # Scheduling
    max_queue_size: int = 2048
    max_decode_batch_size: int = 16
    max_num_batched_tokens: int = 1024
    prefill_chunk_size: int = 128
    scheduler_sleep_s: float = 0.001

    # Prefix cache
    prefix_cache_max_entries: int = 4096
    prefix_cache_max_tokens: int = 500_000
    prefix_cache_min_tokens: int = 16
    prefix_cache_lengths: List[int] = field(default_factory=lambda: [64, 128, 256, 512])

    # Decode optimizations
    enable_cuda_graph_decode: bool = True
    cuda_graph_batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    enable_triton_rmsnorm: bool = True

    # Runtime defaults
    default_max_new_tokens: int = 128
    default_temperature: float = 0.0
