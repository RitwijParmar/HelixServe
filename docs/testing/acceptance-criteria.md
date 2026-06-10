# Acceptance Criteria

The testing upgrade is accepted when:

- CPU-only tests run with the toy backend.
- Allocator tests include invalid input and exhaustion behavior.
- API tests include both valid completion calls and invalid sampling payloads.
- The test matrix identifies optional accelerator coverage separately from required CI checks.
- Debug notes point maintainers to allocator stats, request payloads, and runtime config.
