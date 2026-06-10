# Observability Checks

Verify these during testing:

- `/healthz` returns `{"status": "ok"}`.
- `/stats` includes backend and allocator/runtime statistics.
- `/metrics` exposes `helixserve_requests_total`.
- Allocator stats include free blocks, live blocks, used tokens, utilization, and memory pressure.

For production incidents, compare memory pressure before and after request release.
