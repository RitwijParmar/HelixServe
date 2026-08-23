from __future__ import annotations

import torch

from kernels.rmsnorm_triton import rms_norm


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    rows = 4096
    cols = 4096
    x = torch.randn((rows, cols), device="cuda", dtype=torch.float16)
    w = torch.randn((cols,), device="cuda", dtype=torch.float16)

    for _ in range(50):
        _ = rms_norm(x, w)
    torch.cuda.synchronize()
    print("profiled rmsnorm")


if __name__ == "__main__":
    main()
