# Test Data

Use short deterministic prompts for default tests:

- `hello`
- `Say hi`
- `system: Be brief.\nuser: Summarize allocator pressure.`

Boundary payloads:

- `max_tokens=0` should fail validation.
- `temperature=2.5` should fail validation.
- `top_k=-1` should fail validation.
- Requests that exceed available KV pages should raise allocator exhaustion without mutating unrelated requests.
