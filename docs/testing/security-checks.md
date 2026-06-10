# Security Checks

HelixServe currently exposes local development endpoints without auth. Testing should still verify:

- Invalid JSON and invalid generation parameters are rejected.
- Streaming responses do not expose internal tracebacks.
- Metrics contain operational counters, not prompt contents.
- Future auth or tenancy controls include negative tests before release.
