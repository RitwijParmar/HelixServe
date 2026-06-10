# CI Strategy

Recommended jobs:

- `python-unit`: run allocator, prefix cache, scheduler, and server tests on Python 3.11.
- `lint-imports`: verify optional CUDA/Triton modules remain import-safe.
- `gpu-optional`: run CUDA/Triton tests only on labeled GPU runners.

Keep CPU CI fast by using `EngineConfig(use_toy_backend=True, device="cpu")` for endpoint tests.
