# HelixServe Benchmark Report Template

## Setup

- Date:
- GPU / VM:
- Driver / CUDA:
- Commit:
- Backend:
- Model:

## Config Snapshot

- `kv_block_size`:
- `kv_total_blocks`:
- `max_decode_batch_size`:
- `prefill_chunk_size`:
- `max_num_batched_tokens`:
- `enable_cuda_graph_decode`:

## Workloads

1. Short prompt / short output
2. Long prompt / short output
3. Mixed prompt lengths
4. Burst arrivals
5. Repeated-prefix prompts

## Results Table

| Variant | Throughput (tok/s) | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | E2E p95 | KV Utilization | Fragmentation | Prefix Hit Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline |  |  |  |  |  |  |  |  |  |
| + Paged KV |  |  |  |  |  |  |  |  |  |
| + Continuous Batching |  |  |  |  |  |  |  |  |  |
| + Chunked Prefill |  |  |  |  |  |  |  |  |  |
| + Prefix Cache |  |  |  |  |  |  |  |  |  |
| + CUDA Graph |  |  |  |  |  |  |  |  |  |
| + Triton Kernel |  |  |  |  |  |  |  |  |  |

## Profiling Summary

- Nsight Systems observation:
- Nsight Compute observation:
- CPU launch overhead delta:
- Top GPU kernels by time:

## Interpretation

- Why contiguous KV was insufficient:
- Where continuous batching helped / hurt:
- Chunked prefill effect on decode latency:
- CUDA Graph effect on decode CPU overhead:
- Triton kernel contribution end-to-end:

## Next Steps

- Multi-GPU roadmap:
- Attention-kernel roadmap:
- Speculative decode roadmap:
