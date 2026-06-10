# Test Matrix

| Area | Test Type | Coverage Target | Priority |
| --- | --- | --- | --- |
| KV allocator | Unit | token append, release, refcount pinning, exhaustion, invalid tokens | P0 |
| Prefix cache | Unit | hit/miss, eviction, pinned block lifecycle | P0 |
| Scheduler | Unit/integration | prefill/decode queues, batching limits, rejected requests | P0 |
| FastAPI completions | Integration | request validation, usage accounting, streaming chunks | P0 |
| Chat compatibility | Integration | message flattening and shared completion path | P1 |
| Metrics | Integration | counters, memory pressure, request totals | P1 |
| CUDA graph decode | Conditional | skip-safe behavior when CUDA is unavailable | P2 |
| Triton kernels | Conditional | numerical sanity and skip-safe imports | P2 |
