#include "include/sequential_scan_kernel.cuh"
#include "include/sequential_scan_kernel_bf16.cuh"
#include <cuda_bf16.h>
#include <torch/types.h>
#include <vector>

torch::Tensor scan_forward(torch::Tensor alpha, torch::Tensor beta) {
  const int numBatch = alpha.size(0);
  const int numContext = alpha.size(1);
  const int dModel = alpha.size(2);

  auto out = torch::empty_like(alpha);

  dim3 blockDim;
  dim3 gridDim(numBatch);

  switch (dModel) {
  case 512:
    blockDim.x = 128;
    sequential_scan_forward<512, 128>
        <<<gridDim, blockDim>>>(alpha.data_ptr<float>(), beta.data_ptr<float>(),
                                out.data_ptr<float>(), numContext);
    break;

  case 768:
    blockDim.x = 192;
    sequential_scan_forward<768, 192>
        <<<gridDim, blockDim>>>(alpha.data_ptr<float>(), beta.data_ptr<float>(),
                                out.data_ptr<float>(), numContext);
    break;

  case 1024:
    blockDim.x = 256;
    sequential_scan_forward<1024, 256>
        <<<gridDim, blockDim>>>(alpha.data_ptr<float>(), beta.data_ptr<float>(),
                                out.data_ptr<float>(), numContext);
    break;

  case 1536:
    blockDim.x = 384;
    sequential_scan_forward<1536, 384>
        <<<gridDim, blockDim>>>(alpha.data_ptr<float>(), beta.data_ptr<float>(),
                                out.data_ptr<float>(), numContext);
    break;

  case 2048:
    blockDim.x = 512;
    sequential_scan_forward<2048, 512>
        <<<gridDim, blockDim>>>(alpha.data_ptr<float>(), beta.data_ptr<float>(),
                                out.data_ptr<float>(), numContext);
    break;

  default:
    throw std::runtime_error("Unsupported value for dimension 2 of tensor.");
  }

  return out;
}

std::vector<torch::Tensor> scan_backward(torch::Tensor alpha_saved,
                                         torch::Tensor h_saved,
                                         torch::Tensor grad_out) {
  const int numBatch = alpha_saved.size(0);
  const int numContext = alpha_saved.size(1);
  const int dModel = alpha_saved.size(2);

  auto grad_alpha = torch::empty_like(alpha_saved);
  auto grad_beta = torch::empty_like(alpha_saved);

  dim3 blockDim;
  dim3 gridDim(numBatch);

  std::vector<torch::Tensor> outputs;

  switch (dModel) {
  case 512:
    blockDim.x = 128;
    sequential_scan_backward<512, 128><<<gridDim, blockDim>>>(
        alpha_saved.data_ptr<float>(), h_saved.data_ptr<float>(),
        grad_out.data_ptr<float>(), grad_alpha.data_ptr<float>(),
        grad_beta.data_ptr<float>(), numContext);
    break;

  case 768:
    blockDim.x = 192;
    sequential_scan_backward<768, 192><<<gridDim, blockDim>>>(
        alpha_saved.data_ptr<float>(), h_saved.data_ptr<float>(),
        grad_out.data_ptr<float>(), grad_alpha.data_ptr<float>(),
        grad_beta.data_ptr<float>(), numContext);
    break;

  case 1024:
    blockDim.x = 256;
    sequential_scan_backward<1024, 256><<<gridDim, blockDim>>>(
        alpha_saved.data_ptr<float>(), h_saved.data_ptr<float>(),
        grad_out.data_ptr<float>(), grad_alpha.data_ptr<float>(),
        grad_beta.data_ptr<float>(), numContext);
    break;

  case 1536:
    blockDim.x = 384;
    sequential_scan_backward<1536, 384><<<gridDim, blockDim>>>(
        alpha_saved.data_ptr<float>(), h_saved.data_ptr<float>(),
        grad_out.data_ptr<float>(), grad_alpha.data_ptr<float>(),
        grad_beta.data_ptr<float>(), numContext);
    break;

  case 2048:
    blockDim.x = 512;
    sequential_scan_backward<2048, 512><<<gridDim, blockDim>>>(
        alpha_saved.data_ptr<float>(), h_saved.data_ptr<float>(),
        grad_out.data_ptr<float>(), grad_alpha.data_ptr<float>(),
        grad_beta.data_ptr<float>(), numContext);
    break;

  default:
    throw std::runtime_error("Unsupported value for dimension 2 of tensor.");
  }

  outputs.push_back(grad_alpha);
  outputs.push_back(grad_beta);

  return outputs;
}

torch::Tensor scan_forward_half(torch::Tensor alpha, torch::Tensor beta) {
  const int numBatch = alpha.size(0);
  const int numContext = alpha.size(1);
  const int dModel = alpha.size(2);

  auto out = torch::empty_like(alpha);

  dim3 blockDim;
  dim3 gridDim(numBatch);

  switch (dModel) {
  case 512:
    blockDim.x = 64;
    sequential_scan_forward_half<512, 64><<<gridDim, blockDim>>>(
        reinterpret_cast<__nv_bfloat16 *>(alpha.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(beta.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(out.data_ptr<at::BFloat16>()),
        numContext);
    break;

  case 768:
    blockDim.x = 96;
    sequential_scan_forward_half<768, 96><<<gridDim, blockDim>>>(
        reinterpret_cast<__nv_bfloat16 *>(alpha.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(beta.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(out.data_ptr<at::BFloat16>()),
        numContext);
    break;

  case 1024:
    blockDim.x = 128;
    sequential_scan_forward_half<1024, 128><<<gridDim, blockDim>>>(
        reinterpret_cast<__nv_bfloat16 *>(alpha.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(beta.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(out.data_ptr<at::BFloat16>()),
        numContext);
    break;

  case 1536:
    blockDim.x = 192;
    sequential_scan_forward_half<1536, 192><<<gridDim, blockDim>>>(
        reinterpret_cast<__nv_bfloat16 *>(alpha.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(beta.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(out.data_ptr<at::BFloat16>()),
        numContext);
    break;

  case 2048:
    blockDim.x = 256;
    sequential_scan_forward_half<2048, 256><<<gridDim, blockDim>>>(
        reinterpret_cast<__nv_bfloat16 *>(alpha.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(beta.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(out.data_ptr<at::BFloat16>()),
        numContext);
    break;

  default:
    throw std::runtime_error("Unsupported value for dimension 2 of tensor.");
  }

  return out;
}

std::vector<torch::Tensor> scan_backward_half(torch::Tensor alpha_saved,
                                              torch::Tensor h_saved,
                                              torch::Tensor grad_out) {
  const int numBatch = alpha_saved.size(0);
  const int numContext = alpha_saved.size(1);
  const int dModel = alpha_saved.size(2);

  auto grad_alpha = torch::empty_like(alpha_saved);
  auto &grad_beta = h_saved;

  dim3 blockDim;
  dim3 gridDim(numBatch);

  std::vector<torch::Tensor> outputs;

  switch (dModel) {
  case 512:
    blockDim.x = 64;
    sequential_scan_backward_half<512, 64><<<gridDim, blockDim>>>(
        reinterpret_cast<__nv_bfloat16 *>(alpha_saved.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(h_saved.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(grad_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(grad_alpha.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(grad_beta.data_ptr<at::BFloat16>()),
        numContext);
    break;

  case 768:
    blockDim.x = 96;
    sequential_scan_backward_half<768, 96><<<gridDim, blockDim>>>(
        reinterpret_cast<__nv_bfloat16 *>(alpha_saved.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(h_saved.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(grad_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(grad_alpha.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(grad_beta.data_ptr<at::BFloat16>()),
        numContext);
    break;

  case 1024:
    blockDim.x = 128;
    sequential_scan_backward_half<1024, 128><<<gridDim, blockDim>>>(
        reinterpret_cast<__nv_bfloat16 *>(alpha_saved.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(h_saved.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(grad_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(grad_alpha.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(grad_beta.data_ptr<at::BFloat16>()),
        numContext);
    break;

  case 1536:
    blockDim.x = 192;
    sequential_scan_backward_half<1536, 192><<<gridDim, blockDim>>>(
        reinterpret_cast<__nv_bfloat16 *>(alpha_saved.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(h_saved.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(grad_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(grad_alpha.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(grad_beta.data_ptr<at::BFloat16>()),
        numContext);
    break;

  case 2048:
    blockDim.x = 256;
    sequential_scan_backward_half<2048, 256><<<gridDim, blockDim>>>(
        reinterpret_cast<__nv_bfloat16 *>(alpha_saved.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(h_saved.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(grad_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(grad_alpha.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(grad_beta.data_ptr<at::BFloat16>()),
        numContext);
    break;

  default:
    throw std::runtime_error("Unsupported value for dimension 2 of tensor.");
  }

  outputs.push_back(grad_alpha);
  outputs.push_back(grad_beta);

  return outputs;
}