# GCP L4 shared-prefix benchmark

## Decision

Use native SGLang for this workload. The fixed and adaptive TickYantra admission windows reduced throughput and produced multi-second queueing tails. The run is retained as a falsifiable negative result and as the evidence that triggered two controller corrections.

## Environment

| Field | Value |
|---|---|
| Project commit under test | `ab8b2b0` |
| GCP machine | `g2-standard-8`, `us-central1-b` |
| GPU | 1 × NVIDIA L4, 23,034 MiB |
| Driver | 580.173.02 |
| SGLang | v0.5.16, image digest `sha256:7b6a35df9839fd593a94a1eaee82d7777f472225d9f3ad1f8a2e0cb2bd1785d0` |
| Model | `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`, BF16 |
| Cache | SGLang RadixCache enabled; official native runner reported 99.5% prompt-token cache hit |
| Workload | 100 requests, concurrency 16, 10 ordered shared-prefix groups, 128 system tokens, 64 question tokens, 64 output tokens, seed 17, 8 warm-ups |

## Valid results

| Variant | Success | Throughput | Output rate | p50 / p95 / p99 TTFT | p95 ITL | p95 E2E |
|---|---:|---:|---:|---:|---:|---:|
| Official SGLang native | 100/100 | **3.70 req/s** | **236.75 tok/s** | 123.54 / **137.93** / 178.70 ms | 59.62 ms | **3,874.09 ms** |
| TickYantra native proxy | 100/100 | 3.55 req/s | 227.39 tok/s | 296.53 / 472.78 / 476.29 ms | 59.31 ms | 4,177.12 ms |
| TickYantra static, limit 12 | 100/100 | 2.90 req/s | 185.77 tok/s | 128.55 / 3,939.72 / 3,996.37 ms | 59.14 ms | 7,631.61 ms |
| TickYantra adaptive, initial 12 | 100/100 | 2.90 req/s | 185.66 tok/s | 127.06 / 3,960.67 / 3,995.49 ms | 59.24 ms | 7,672.68 ms |

The official SGLang runner is the primary native baseline. The repository's black-box client is retained as a cross-check and captures raw per-request rows. Differences between the two native measurements reflect different client implementations and request accounting, so they are not presented as a proxy-overhead estimate.

## What the run disproved

At concurrency 16, a gateway window of 12 forced later requests to wait roughly one generation wave before admission. Client TTFT therefore reached about four seconds at p95 even though inter-token latency stayed near 59 ms. The adaptive controller then increased its limit from 12 to 13 because its feedback clock began after admission and excluded that queue wait. The affinity fingerprint also used the first 1,024 characters, which included unique question text; `/stats` consequently reported 100 hot prefixes rather than grouping the shared context.

The follow-up fix measures TTFT from gateway arrival and shortens the default fingerprint to 128 characters. These corrections are not retroactively credited with better performance. A future controller must be rerun and must beat or match the official native baseline without increasing failures before it can be recommended.

## Invalid runner artifact

`official-sglang.jsonl` is preserved but excluded. SGLang's OpenAI benchmark client raised `IndexError` on empty SSE choice frames and counted only 13 of 100 requests. `official-sglang-native.jsonl` uses the native SGLang backend and completed all 100 requests, so it is the valid official result.

## Raw evidence

- `official-sglang-native.jsonl` — valid official native run, including per-request details and cache report.
- `native.json` — repository black-box client against SGLang directly.
- `static.json`, `static-stats.json` — fixed admission window and final controller state.
- `adaptive.json`, `adaptive-stats.json` — adaptive admission and final controller state.
- `official-sglang.jsonl` — invalid OpenAI-runner attempt, retained for failure transparency.

The GPU VM was stopped after collection. Its two-hour shutdown guard remains part of provisioning as a second cost boundary.
