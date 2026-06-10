# STLC Plan

## Scope

Validate that HelixServe accepts OpenAI-style completion requests, schedules inference safely, manages KV cache pages without leaks, and exposes actionable health, stats, and Prometheus metrics.

## Phases

1. Requirement analysis: confirm API contracts, allocator constraints, scheduler backpressure, and optional accelerator behavior.
2. Test planning: prioritize allocator correctness, request validation, queue safety, prefix cache reuse, and endpoint compatibility.
3. Test design: create unit tests for pure components and integration tests around the FastAPI app with the toy backend.
4. Environment setup: run CPU toy-backend tests by default; keep CUDA/Triton tests optional and skip-safe.
5. Execution: run targeted pytest suites before PR review and full suite before merge.
6. Defect closure: link each bug to a regression test and note the smallest failing component.

## Exit Criteria

- Core allocator, scheduler, prefix cache, and server endpoint tests pass.
- Negative API validation paths are covered.
- Optional GPU tests are documented and isolated from CPU CI.
