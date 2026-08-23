from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


def rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x * rms * weight


if triton is not None:
    @triton.jit
    def _rms_norm_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        x = tl.load(x_ptr + row * stride_xm + cols * stride_xn, mask=mask, other=0.0)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0)

        var = tl.sum(x * x, axis=0) / n_cols
        inv_rms = tl.rsqrt(var + eps)
        y = x * inv_rms * w

        tl.store(y_ptr + row * stride_ym + cols * stride_yn, y, mask=mask)


def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton not available")
    if not x.is_cuda:
        raise ValueError("x must be CUDA tensor for Triton kernel")
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    if weight.ndim != 1:
        raise ValueError("weight must be 1D")
    if x.shape[1] != weight.shape[0]:
        raise ValueError("weight shape must match last dim of x")

    x = x.contiguous()
    weight = weight.contiguous()
    rows, cols = x.shape

    y = torch.empty_like(x)

    block = triton.next_power_of_2(cols)
    block = min(max(block, 128), 4096)

    _rms_norm_kernel[(rows,)](
        x,
        weight,
        y,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        cols,
        eps,
        BLOCK_SIZE=block,
        num_warps=4,
    )
    return y


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if triton is not None and x.is_cuda:
        return rms_norm_triton(x, weight, eps)
    return rms_norm_reference(x, weight, eps)


def benchmark_once(
    *,
    rows: int = 4096,
    cols: int = 4096,
    dtype: torch.dtype = torch.float16,
) -> Tuple[float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton benchmark")

    x = torch.randn((rows, cols), device="cuda", dtype=dtype)
    w = torch.randn((cols,), device="cuda", dtype=dtype)

    for _ in range(10):
        rms_norm_reference(x, w)
        rms_norm_triton(x, w)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(50):
        y_ref = rms_norm_reference(x, w)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / 50.0

    start.record()
    for _ in range(50):
        y_tri = rms_norm_triton(x, w)
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / 50.0

    max_diff = (y_ref - y_tri).abs().max().item()
    if max_diff > 1e-2:
        raise RuntimeError(f"numerical mismatch too high: {max_diff}")

    return torch_ms, triton_ms
