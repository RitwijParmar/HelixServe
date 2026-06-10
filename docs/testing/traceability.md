# Traceability

| Requirement | Code Surface | Test Surface |
| --- | --- | --- |
| Reject invalid generation parameters | `server.main.CompletionRequest` | `tests/test_server.py` |
| Produce completion responses with usage | `server.main.create_app` | `tests/test_server.py` |
| Preserve KV block ownership | `cache.allocator.KVBlockAllocator` | `tests/test_allocator.py` |
| Avoid cache leaks after pinned prefixes | `cache.allocator`, `cache.prefix_cache` | `tests/test_allocator.py`, `tests/test_prefix_cache.py` |
| Batch decode without starving prefill | `engine.scheduler` | `tests/test_engine_scheduler.py` |
| Export Prometheus metrics | `server.main.metrics` | `tests/test_server.py` |
