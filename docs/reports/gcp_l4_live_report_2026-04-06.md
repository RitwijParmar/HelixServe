# HelixServe Live Benchmark Report (GCP L4)

## Setup

- Date: 2026-04-06
- Project: `project-2281c357-4539-4bc6-b96`
- VM: `helixserve-g2` (`g2-standard-4`, 1x NVIDIA L4)
- Public endpoint: `http://34.31.35.113:8000`
- Driver / CUDA (host): `580.126.09` / `13.0`
- Container base CUDA runtime: `12.4.1`
- Commit: `e874e05` + benchmark/report scripts
- Backend: `ToyDecoderBackend` on `cuda`
- Model config name: `sshleifer/tiny-gpt2` (toy mode enabled)

Raw artifacts:

- [`docs/results/20260406T110454Z`](/Users/ritwij/Documents/HelixServe/docs/results/20260406T110454Z)
- [`docs/results/20260406T110729Z`](/Users/ritwij/Documents/HelixServe/docs/results/20260406T110729Z)

## Runtime Config Snapshot

- `kv_block_size`: `16`
- `kv_total_blocks`: `4096`
- `max_decode_batch_size`: `16`
- `prefill_chunk_size`: `128`
- `max_num_batched_tokens`: `1024`
- `enable_cuda_graph_decode`: `true`

## Workload Results

| Workload | Requests | Concurrency | Throughput (tok/s) | TTFT p50 (s) | TTFT p95 (s) | E2E p95 (s) |
|---|---:|---:|---:|---:|---:|---:|
| Short | 200 | 16 | 1110.75 | 0.132 | 0.202 | 0.390 |
| Long | 200 | 16 | 886.58 | 0.301 | 0.317 | 0.503 |
| Mixed | 200 | 16 | 1021.13 | 0.182 | 0.216 | 0.406 |
| Repeated Prefix | 200 | 16 | 1149.81 | 0.135 | 0.154 | 0.346 |
| Burst (mixed) | 400 | 64 | 1287.51 | 0.402 | 0.554 | 1.361 |

## Memory And Cache Observations

- Prefix cache in suite run `20260406T110729Z`:
  - before: `lookups=1201`, `hits=1188`, `hit_rate=0.9892`
  - after: `lookups=2401`, `hits=2388`, `hit_rate=0.9946`
- KV allocator after suite:
  - `live_blocks=92`
  - `used_tokens=1472`
  - `memory_pressure=0.0225`
  - `internal_waste_tokens=0`

## Notes

- Repeated-prefix workload improved TTFT and throughput relative to mixed/short.
- Long prompts produced the highest TTFT and p95 latency as expected.
- High-concurrency burst increased throughput but raised tail latency.
