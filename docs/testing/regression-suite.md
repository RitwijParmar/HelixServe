# Regression Suite

Run before merging runtime or API changes:

```bash
pytest -q tests/test_allocator.py tests/test_prefix_cache.py tests/test_engine_scheduler.py tests/test_server.py
```

Run before release candidates:

```bash
pytest -q
```

For CUDA-enabled environments, include the CUDA and Triton tests and record device details in the PR.
