# HelixServe

HelixServe is a mini LLM serving engine focused on runtime internals, not chatbot product surface.

## Language Split

HelixServe uses a hybrid implementation:

- Python for server, scheduler, allocator, orchestration, and benchmarking
- Triton for custom GPU kernel work (`kernels/rmsnorm_triton.py`)
- CUDA C++ for a focused low-level op (`cuda_ext/csrc/cu_seqlens*`)

## Scope (v1)

This project implements the six locked features:

1. Decoder-only model backend on one GPU (`ToyDecoderBackend` by default, optional HF backend).
2. Paged KV-cache allocator with fixed-size blocks.
3. Continuous batching scheduler.
4. Split prefill/decode with chunked prefill.
5. CUDA Graph replay on steady decode path (Toy backend on CUDA).
6. One custom Triton kernel (`kernels/rmsnorm_triton.py`).

## Repository Layout

- `server/` HTTP API and streaming.
- `engine/` runtime config, scheduler, request lifecycle, CUDA graph helper.
- `cache/` paged allocator and prefix cache.
- `model/` model backends and tokenizer.
- `kernels/` Triton kernel and kernel benchmark.
- `cuda_ext/` optional CUDA C++ extension for decode-time `cu_seqlens` building.
- `bench/` load generation and benchmark runner.
- `metrics/` Prometheus metrics registry.
- `deploy/` Dockerfile and GCP deployment scripts.
- `profiling/` Nsight helper scripts.
- `docs/` architecture and execution plan.
- `tests/` unit and async integration tests.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Optional CUDA C++ extension build (Linux + CUDA toolkit):

```bash
HELIX_BUILD_CUDA_EXT=1 pip install -e .[dev]
```

Run server:

```bash
HELIX_USE_TOY_BACKEND=1 HELIX_DEVICE=cuda python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Send request:

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H "content-type: application/json" \
  -d '{"prompt":"Explain paged KV-cache", "max_tokens":32}' | jq
```

Streaming request:

```bash
curl -N http://127.0.0.1:8000/v1/completions \
  -H "content-type: application/json" \
  -d '{"prompt":"Explain chunked prefill", "max_tokens":32, "stream":true}'
```

## Benchmark

```bash
python -m bench.run_benchmark --url http://127.0.0.1:8000 --requests 200 --concurrency 16 --mode mixed --max-tokens 64 --stream
```

Suggested workloads:

- `--mode short`
- `--mode long`
- `--mode mixed`
- `--mode repeated_prefix`

Run the full live suite (short/long/mixed/repeated-prefix/burst) and save artifacts:

```bash
python -m bench.run_live_suite --url http://127.0.0.1:8000 --stream
```

## Triton Kernel Benchmark

```bash
python -m kernels.benchmark_rmsnorm --rows 4096 --cols 4096
```

## Profiling

Nsight Systems decode capture:

```bash
bash profiling/nsys_decode_capture.sh helixserve_decode
```

Nsight Compute kernel capture:

```bash
bash profiling/ncu_rmsnorm.sh helixserve_rmsnorm
```

Nsight Compute CUDA C++ extension capture:

```bash
bash profiling/ncu_cu_seqlens.sh helixserve_cu_seqlens
```

## Metrics

- Prometheus endpoint: `GET /metrics`
- Engine stats endpoint: `GET /stats`

Key runtime metrics:

- TTFT, ITL, E2E latency histograms
- Throughput counters
- KV utilization and fragmentation
- Queue depth, active decode batch size, and active decode batched tokens
- Prefix cache hit rate

## GCP Deployment

Create G2/L4 VM (Deep Learning VM image family):

```bash
PROJECT_ID=<your-project> ZONE=us-central1-c bash deploy/gcp_create_g2_l4.sh
```

Deploy container to VM:

```bash
PROJECT_ID=<your-project> INSTANCE_NAME=<vm-name> bash deploy/deploy_to_gcp_vm.sh
```

Configure VM idle auto-shutdown (default: 30 minutes idle):

```bash
PROJECT_ID=<your-project> INSTANCE_NAME=<vm-name> bash deploy/setup_idle_shutdown_on_vm.sh
```

Disable auto-shutdown temporarily on the VM:

```bash
sudo touch /var/lib/helixserve/disable_idle_shutdown
```

Re-enable auto-shutdown:

```bash
sudo rm -f /var/lib/helixserve/disable_idle_shutdown
```

## Tests

```bash
pytest -q
```
