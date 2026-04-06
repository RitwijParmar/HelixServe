# HelixServe Architecture

## Core Loop

`ContinuousBatchScheduler` runs the runtime loop:

1. Admit requests from inbound queue.
2. Attempt prefix-cache attach/state-restore.
3. Decode-first scheduling pass (`_decode_step`).
4. Chunked prefill pass under token budget (`_prefill_step`).
5. Publish metrics.

## Components

### `cache/allocator.py`

- `KVBlockAllocator`
- Fixed-size block paging for KV capacity accounting.
- Request attach/release and prefix pin/unpin refcount semantics.
- Stats: utilization, internal waste, memory pressure.

### `cache/prefix_cache.py`

- `PrefixCache`
- LRU cache of prompt prefix hash -> block table snapshot.
- Optional backend state payload reuse.
- Stats: entries, lookups, hits, hit rate.

### `engine/scheduler.py`

- `ContinuousBatchScheduler`
- Continuous batching with request-policy grouping.
- Prefill/decode split with chunked prefill.
- Backpressure and rejection on queue/memory pressure.
- Decode-time `cu_seqlens` build call (via optional CUDA C++ extension).

### `engine/cuda_graph.py`

- `CUDAGraphDecodeCache`
- Captures static decode step for fixed batch sizes.
- Replays captured graph for lower launch overhead.

### `model/toy_backend.py`

- `ToyDecoderBackend` (default)
- Tiny decoder-only GRU LM with per-request state.
- Batch decode and optional CUDA Graph replay.

### `model/hf_backend.py`

- `TransformersDecoderBackend`
- Hugging Face fallback backend for real model inference.

### `kernels/rmsnorm_triton.py`

- Fused RMSNorm Triton kernel.
- Numerical parity check vs reference implementation.
- Isolated kernel benchmark utility.

### `cuda_ext/csrc/cu_seqlens*`

- Optional CUDA C++ extension (`helix_cuda_ext`) exposed to Python.
- Implements `build_cu_seqlens(lengths)` for decode batch prep.
- Used by scheduler on CUDA paths, with Python fallback when extension isn't built.

## Request Lifecycle

1. Request arrives with prompt and generation params.
2. Prompt tokenized and queued.
3. Prefix lookup attempts KV + backend-state reuse.
4. Remaining prompt prefills in chunks.
5. Request joins decode queue.
6. Decode emits one token per scheduler step.
7. Completion triggers cleanup and KV release.

## Metrics

- `TTFT`, `ITL`, request latency histograms.
- Queue depth and active decode batch size.
- KV used tokens, utilization, fragmentation.
- Prefix cache entries/hit rate.

## Design Notes

- KV pages are logical allocator pages in v1; they are not yet consumed by custom attention kernels.
- Prefix correctness is preserved by sealing prompt page before first decode token.
- Decode is prioritized over prefill to protect inter-token latency under mixed load.
