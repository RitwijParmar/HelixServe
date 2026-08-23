# TickYantra

**SLO-aware inference control for latency-sensitive AI systems.**

[Live control-lab demo](https://tickyantra-control-lab.ritwij.chatgpt.site/) · [Narrated end-to-end video](docs/assets/demo/tickyantra/final/tickyantra_end_to_end_demo.mp4) · [Measured GCP L4 report](docs/results/v2-gcp-l4/report.md)

TickYantra—*tick* for the atomic event in an electronic market, *yantra* for a precise machine—is an end-to-end serving project built for the failure modes that matter in low-latency production: tail latency, overload, prefix-heavy traffic, measurement integrity, and cost-bounded GPU deployment.

It does not pretend a toy decoder is a production engine. [SGLang v0.5.16](https://github.com/sgl-project/sglang) owns the real CUDA data plane: RadixAttention, continuous batching, paged KV memory, and model execution. TickYantra adds an inspectable SLO control plane and a reproducible experiment layer around it.

## Why this is different

- **No toy path or silent fallback.** Readiness requires a live SGLang server; failures remain failures.
- **Tail-latency control.** A bounded admission window adjusts from rolling p95 TTFT instead of maximizing queue depth blindly.
- **Cache-aware fairness.** Recently served prompt prefixes get affinity until a request approaches its queue deadline, then age wins.
- **Honest measurement.** The benchmark records true streamed TTFT, every inter-token/chunk interval, E2E latency, status, errors, and raw per-request rows.
- **Comparable experiments.** Native SGLang, fixed-window, and adaptive variants share the same model, seed, prompts, and output lengths.
- **Real deployment posture.** SGLang is pinned, the model port binds only to loopback, metrics are first-class, and a GCP cost guard stops the GPU VM after two hours.

## Architecture

```text
OpenAI client
     |
     v
TickYantra :8000 ----> /stats + /metrics
     |
     | bounded admission + prefix affinity + SLO feedback
     v
SGLang :30000 ----> RadixAttention + continuous batching + paged KV
     |
     v
NVIDIA L4
```

TickYantra transparently supports `/v1/models`, `/v1/completions`, and `/v1/chat/completions`, including streaming. See [the architecture note](docs/v2/architecture.md) for the lifecycle and correctness boundaries.

## Run locally

The control plane can be tested without a GPU; actual generation always requires SGLang.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
ruff check .
```

With SGLang already listening on port 30000:

```bash
TICKYANTRA_MODE=adaptive \
TICKYANTRA_TARGET_TTFT_MS=450 \
tickyantra
```

```bash
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","prompt":"Assess this order-flow imbalance","max_tokens":64,"stream":true}'
```

## Run the real GPU stack

Docker Compose starts SGLang v0.5.16 with `Qwen/Qwen2.5-7B-Instruct` and the TickYantra gateway. Neither service is exposed beyond loopback by default.

```bash
docker compose -f deploy/compose.gpu.yaml up -d --build
curl http://127.0.0.1:8000/readyz
```

GCP provisioning uses a single L4 Deep Learning VM and installs a two-hour hard shutdown guard:

```bash
PROJECT_ID=your-project ZONE=us-central1-a bash deploy/provision_gcp.sh
```

The deployment intentionally fails when GPU quota, drivers, or SGLang are unavailable. It never replaces the requested model with fabricated output.

The stack was redeployed and verified on `tickyantra-l4` in `us-central1-b`: the NVIDIA L4 reported 23,034 MiB, `/readyz` confirmed the SGLang upstream, and a live Qwen2.5-7B completion returned 32 tokens. Inference remains loopback-only and the shutdown guard is re-armed on every boot.

## Benchmark

```bash
tickbench \
  --url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --requests 200 \
  --concurrency 16 \
  --max-tokens 64 \
  --repeated-prefix \
  --output docs/results/v2/adaptive-prefix.json
```

For the independent native baseline, use SGLang's official runner directly against port 30000. The full rules—including warm-up, workload parity, failure accounting, and required metadata—are in [the benchmark methodology](docs/v2/benchmark-methodology.md).

## Operational endpoints

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Gateway process health |
| `GET /readyz` | Verifies the real SGLang upstream |
| `GET /stats` | Active limit, queue depth, admissions, rejections, hot prefixes |
| `GET /metrics` | Prometheus counters, gauges, TTFT, queue wait, and E2E histograms |

Optional `TICKYANTRA_API_KEY` protects the gateway. Prefix identity is stored only as a truncated BLAKE2s digest; raw prompts never become metric labels.

## Repository map

| Path | What it proves |
|---|---|
| `tickyantra/` | SLO controller, prefix-aware queue, streaming reverse proxy, metrics |
| `bench/` | Seeded black-box benchmark with raw latency evidence |
| `deploy/` | Pinned SGLang stack and bounded-cost GCP provisioning |
| `tests/` | Async concurrency, timeout, adaptation, streaming, auth, readiness |
| `docs/v2/` | Design boundaries and experimental methodology |
| `legacy/v0_toy/` | Preserved, explicitly non-production v0 experiments |

## Measured result: native wins this burst

A real `g2-standard-8` / NVIDIA L4 run served 100 shared-prefix requests with 16-way concurrency, 128-token system context, 64-token questions, and 64-token outputs. All three valid variants completed 100/100 requests.

| Path | Throughput | p95 TTFT | p95 E2E | Decision |
|---|---:|---:|---:|---|
| Official SGLang native runner | **3.70 req/s** | **137.93 ms** | **3,874.09 ms** | Ship for this workload |
| TickYantra native proxy | 3.55 req/s | 472.78 ms | 4,177.12 ms | Measurement cross-check |
| TickYantra static window (12) | 2.90 req/s | 3,939.72 ms | 7,631.61 ms | Do not ship |
| TickYantra adaptive (12→13) | 2.90 req/s | 3,960.67 ms | 7,672.68 ms | Do not ship; feedback bug found |

The fixed and adaptive gateways amplified queueing under this saturated burst. That is a useful negative result, not an improvement claim: native SGLang is the deployment choice until a controller clears the native tail-latency gate. The experiment also exposed two defects now covered by the follow-up fix: adaptive feedback excluded queue time, and a 1,024-character affinity key fragmented the shared prefix into 100 keys. See the [full report and raw artifacts](docs/results/v2-gcp-l4/report.md).

## License

MIT
