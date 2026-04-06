# HelixServe Weekly Task Board

## Week 1: Foundation + Baseline

Goal: runnable single-node baseline.

Tasks:

- [ ] Provision GCP G2/L4 VM and verify `nvidia-smi`.
- [ ] Bring up server (`server/main.py`) with toy backend.
- [ ] Validate end-to-end `/v1/completions` and streaming.
- [ ] Add CI: `ruff`, `pytest`.

Primary files/classes:

- `server/main.py`
- `engine/runtime.py` (`HelixEngine`)
- `model/toy_backend.py` (`ToyDecoderBackend`)

Codex prompts:

1. "Add GitHub Actions workflow to run ruff and pytest on push."
2. "Add smoke test for /healthz and /v1/completions using FastAPI TestClient."

## Week 2: Paged KV + Prefix Cache

Goal: KV memory lifecycle and prefix reuse.

Tasks:

- [ ] Tune `KVBlockAllocator` block size/total blocks for L4 memory budget.
- [ ] Add allocator stress tests and fragmentation assertions.
- [ ] Validate prefix cache hit-rate under repeated-prefix workload.

Primary files/classes:

- `cache/allocator.py` (`KVBlockAllocator`)
- `cache/prefix_cache.py` (`PrefixCache`)
- `engine/scheduler.py` (`_apply_prefix_hit`, `_mark_prefill_complete`)

Codex prompts:

1. "Create stress test that submits thousands of random allocate/release operations and checks no leaked blocks."
2. "Add benchmark script for prefix-cache hit/miss TTFT comparison."

## Week 3: Continuous Batching + Chunked Prefill

Goal: scheduler behavior under mixed load.

Tasks:

- [ ] Tune decode-first policy and prefill token budget.
- [ ] Add queue-delay and active-batch instrumentation checks.
- [ ] Demonstrate long-prefill non-starvation in benchmark report.

Primary files/classes:

- `engine/scheduler.py` (`_decode_step`, `_prefill_step`)
- `bench/run_benchmark.py`
- `metrics/registry.py`

Codex prompts:

1. "Add scheduler trace logging (JSON lines) for admit/prefill/decode/finish events with timestamps."
2. "Add mixed workload benchmark mode with burst arrivals and output p50/p95/p99 TTFT + latency."

## Week 4: CUDA Graph + Triton Kernel

Goal: optimized decode path and one custom kernel.

Tasks:

- [ ] Capture/replay decode graph for batch sizes {1,2,4,8,16}.
- [ ] Benchmark CPU overhead and per-token latency with/without graph.
- [ ] Tune RMSNorm Triton kernel launch params on L4.
- [ ] Build and validate CUDA C++ `cu_seqlens` extension on the target GPU VM.

Primary files/classes:

- `engine/cuda_graph.py` (`CUDAGraphDecodeCache`)
- `model/toy_backend.py` (`decode_batch`)
- `kernels/rmsnorm_triton.py`
- `cuda_ext/csrc/cu_seqlens.cpp`
- `cuda_ext/csrc/cu_seqlens_kernel.cu`

Codex prompts:

1. "Add benchmark that compares decode throughput with CUDA graph enabled vs disabled."
2. "Add Triton kernel unit test for fp16/bf16 with max abs error threshold."

## Week 5: Profiling + Final Report

Goal: publish-quality measurements and narrative.

Tasks:

- [ ] Capture Nsight Systems and Nsight Compute traces.
- [ ] Produce baseline-vs-optimized comparison table.
- [ ] Finalize architecture and tradeoff write-up.

Primary files/classes:

- `profiling/nsys_decode_capture.sh`
- `profiling/ncu_rmsnorm.sh`
- `docs/benchmark_report_template.md`

Codex prompts:

1. "Generate markdown summary table from benchmark JSON files in bench/results/."
2. "Draft final report section: what improved TTFT vs what improved throughput."
