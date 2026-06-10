# Debugging Playbook

1. Reproduce with the toy backend first.
2. Capture request JSON and `EngineConfig`.
3. Check `/stats` before and after the failing request.
4. Inspect allocator `free_blocks`, `live_blocks`, and `used_tokens`.
5. Run the smallest focused test file, then the full regression suite.
6. If only accelerator tests fail, verify CUDA availability and Triton import state.
