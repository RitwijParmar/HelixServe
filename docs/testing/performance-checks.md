# Performance Checks

Performance testing should track:

- Queue wait time under concurrent submissions.
- Decode batch size at saturation.
- KV block memory pressure.
- Prefix cache hit ratio for repeated prompts.
- Completion latency for non-streaming and streaming calls.

Use the toy backend for behavioral load tests and real models only for benchmark runs.
