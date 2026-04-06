import pytest
import torch

from kernels.rmsnorm_triton import rms_norm_reference

try:
    from kernels.rmsnorm_triton import rms_norm_triton
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_rmsnorm_triton_matches_reference() -> None:
    x = torch.randn((256, 1024), device="cuda", dtype=torch.float16)
    w = torch.randn((1024,), device="cuda", dtype=torch.float16)

    y_ref = rms_norm_reference(x, w)
    y_tri = rms_norm_triton(x, w)

    max_diff = (y_ref - y_tri).abs().max().item()
    assert max_diff < 1e-2
