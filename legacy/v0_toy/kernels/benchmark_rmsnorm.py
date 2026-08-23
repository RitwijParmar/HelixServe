from __future__ import annotations

from argparse import ArgumentParser

from kernels.rmsnorm_triton import benchmark_once


def main() -> None:
    parser = ArgumentParser(description="Benchmark Triton RMSNorm kernel")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    args = parser.parse_args()

    torch_ms, triton_ms = benchmark_once(rows=args.rows, cols=args.cols)
    speedup = torch_ms / triton_ms if triton_ms > 0 else 0.0

    print(f"torch_ms={torch_ms:.4f}")
    print(f"triton_ms={triton_ms:.4f}")
    print(f"speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
