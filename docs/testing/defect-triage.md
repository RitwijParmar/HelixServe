# Defect Triage

## Severity

- S0: request corruption, leaked KV blocks, server crash, or unsafe response contract.
- S1: incorrect validation, bad usage accounting, scheduler starvation, or broken metrics.
- S2: optional accelerator skip behavior, documentation mismatch, or noisy logs.

## Triage Notes

Capture the failing request payload, engine configuration, allocator stats before and after the request, and whether the toy backend or accelerator backend was active.
