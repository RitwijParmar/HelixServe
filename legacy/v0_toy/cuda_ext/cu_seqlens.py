from __future__ import annotations

import torch

try:
    import helix_cuda_ext as _cuda_ext
except Exception:  # pragma: no cover - extension is optional
    _cuda_ext = None


def has_compiled_cuda_ext() -> bool:
    return _cuda_ext is not None


def build_cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    """
    Build cu_seqlens (length B+1) from per-sequence lengths (length B).

    Uses the optional CUDA C++ extension when available and falls back to
    torch.cumsum otherwise.
    """
    if lengths.ndim != 1:
        raise ValueError("lengths must be 1D")

    lengths_i32 = lengths.to(dtype=torch.int32)

    if lengths_i32.is_cuda and _cuda_ext is not None:
        return _cuda_ext.build_cu_seqlens(lengths_i32.contiguous())

    out = torch.zeros((lengths_i32.numel() + 1,), dtype=torch.int32, device=lengths_i32.device)
    if lengths_i32.numel() > 0:
        out[1:] = torch.cumsum(lengths_i32, dim=0, dtype=torch.int32)
    return out
