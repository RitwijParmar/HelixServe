# Product Voiceover Brief

Purpose:

- Keep narration product-centric and operator-focused.
- Avoid first-person phrasing (`I`, `me`, `my`) in the final cut.

Narration style:

- Humble and practical.
- Explain tradeoffs with real metrics.
- Prefer concrete behavior examples over hype.

Required references in the spoken track:

- Baseline failure mode: contiguous KV + request blocking + decode overhead.
- Runtime capabilities: paged KV, continuous batching, chunked prefill, prefix cache, CUDA Graph, Triton + CUDA C++ path.
- Measured outcome: baseline to optimized throughput and ITL improvement.
