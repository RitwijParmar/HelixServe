#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

torch::Tensor build_cu_seqlens_cuda(torch::Tensor lengths) {
  const auto seq_count = lengths.size(0);
  auto out = torch::zeros({seq_count + 1}, lengths.options());

  if (seq_count == 0) {
    return out;
  }

  auto* in_ptr = lengths.data_ptr<int32_t>();
  auto* out_ptr = out.data_ptr<int32_t>();

  thrust::device_ptr<int32_t> in_dev(in_ptr);
  thrust::device_ptr<int32_t> out_dev(out_ptr + 1);
  auto stream = at::cuda::getDefaultCUDAStream();

  thrust::inclusive_scan(
      thrust::cuda::par.on(stream.stream()),
      in_dev,
      in_dev + seq_count,
      out_dev);

  return out;
}
