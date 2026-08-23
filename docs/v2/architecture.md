# TickYantra architecture

TickYantra is deliberately a control plane, not a second-rate model runtime. SGLang owns tokenization, RadixAttention prefix reuse, continuous batching, paged KV memory, and CUDA execution. TickYantra owns admission, queue deadlines, prefix-affinity ordering, SLO feedback, and measurement.

```text
OpenAI client -> TickYantra :8000 -> SGLang :30000 -> NVIDIA L4
                  |                  |
                  | /stats          + native scheduler + RadixAttention
                  + /metrics
```

## Request lifecycle

1. The gateway extracts a stable hash from the prompt prefix; raw prompts never enter metrics.
2. Native mode passes through immediately. Static and adaptive modes enforce a bounded concurrency window.
3. Queued requests prefer a recently served prefix until 25% of the queue deadline, after which FIFO age wins.
4. Streaming bytes pass through without buffering. The gateway measures first meaningful SSE data and end-to-end latency.
5. On completion, adaptive mode uses rolling p95 TTFT to add or remove one concurrency slot within configured bounds.

The adaptive algorithm is intentionally small and inspectable. It cannot alter model outputs or SGLang memory state; failure degrades to an HTTP error rather than a fabricated response.

## Correctness boundaries

- No toy backend and no silent fallback exist in v2.
- `/readyz` requires the real SGLang `/health` endpoint.
- Upstream transport errors return `502`; queue deadline expiry returns `429`.
- Raw v0 experiments are preserved under `legacy/v0_toy/` and excluded from the production package.
- SGLang is pinned to v0.5.16 for repeatability.
