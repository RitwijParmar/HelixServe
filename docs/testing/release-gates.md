# Release Gates

Do not release if:

- Allocator release leaks live blocks.
- Request validation lets invalid `max_tokens`, `temperature`, or `top_k` reach the engine.
- `/metrics` or `/healthz` is broken.
- Optional CUDA/Triton tests fail on an environment that claims to support them.

Release notes should include changed runtime config defaults and any benchmark caveats.
