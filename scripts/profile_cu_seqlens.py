from __future__ import annotations

import torch

from cuda_ext import build_cu_seqlens, has_compiled_cuda_ext


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if not has_compiled_cuda_ext():
        print("CUDA C++ extension not loaded; set HELIX_BUILD_CUDA_EXT=1 during install")
        return

    lengths = torch.randint(low=1, high=1024, size=(2048,), device="cuda", dtype=torch.int32)
    for _ in range(100):
        build_cu_seqlens(lengths)
    torch.cuda.synchronize()

    print("profiled cu_seqlens extension")


if __name__ == "__main__":
    main()
