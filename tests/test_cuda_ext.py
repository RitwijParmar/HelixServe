import torch

from cuda_ext import build_cu_seqlens


def test_build_cu_seqlens_cpu_fallback() -> None:
    lengths = torch.tensor([3, 1, 4, 2], dtype=torch.int32)
    cu = build_cu_seqlens(lengths)
    assert cu.dtype == torch.int32
    assert cu.tolist() == [0, 3, 4, 8, 10]


def test_build_cu_seqlens_empty() -> None:
    lengths = torch.tensor([], dtype=torch.int32)
    cu = build_cu_seqlens(lengths)
    assert cu.tolist() == [0]
