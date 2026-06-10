# Risk-Based Testing

Highest risk areas:

- KV allocator reference counting because leaks compound under production traffic.
- Scheduler backpressure because queue overflow can look like random latency spikes.
- API validation because malformed requests should fail before reaching runtime internals.
- Streaming response shape because clients depend on stable chunk semantics.

Lower risk areas:

- Static README examples.
- Optional kernel paths when dependencies are absent and skip markers are working.
