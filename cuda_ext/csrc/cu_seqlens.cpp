#include <torch/extension.h>

#include <vector>

torch::Tensor build_cu_seqlens_cuda(torch::Tensor lengths);

torch::Tensor build_cu_seqlens(torch::Tensor lengths) {
  TORCH_CHECK(lengths.is_cuda(), "lengths must be a CUDA tensor");
  TORCH_CHECK(lengths.dim() == 1, "lengths must be 1D");
  TORCH_CHECK(lengths.scalar_type() == torch::kInt32, "lengths must be int32");

  return build_cu_seqlens_cuda(lengths.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build_cu_seqlens", &build_cu_seqlens, "Build cu_seqlens from per-sequence lengths (CUDA)");
}
