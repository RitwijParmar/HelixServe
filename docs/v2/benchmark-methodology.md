# Benchmark methodology

Every published v2 result must identify the commit, SGLang version, model, GPU, workload, seed, concurrency or arrival rate, request count, prompt/output lengths, and warm-up policy.

The repository harness consumes the streaming OpenAI API and records per-request TTFT, every inter-chunk latency sample, E2E latency, failures, HTTP status, and output-token count. Aggregate reports publish p50/p95/p99 and preserve the raw request rows. The official `sglang.bench_serving` runner is also used against port 30000 as an independent native baseline.

Three variants use the same model and workload:

1. **Native:** direct SGLang, establishing the data-plane ceiling.
2. **Static:** TickYantra with a fixed concurrency window.
3. **Adaptive:** TickYantra with SLO feedback and prefix-affinity queueing.

Runs include unique-prefix and shared-prefix traffic. A claimed improvement is valid only when failure rate does not increase and the raw JSON artifact is committed. Cost guardrails automatically stop the GCP VM after two hours; the final workflow stops it immediately after collection.
