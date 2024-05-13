#include <torch/extension.h>

torch::Tensor scan_forward(torch::Tensor alpha, torch::Tensor beta);
std::vector<torch::Tensor> scan_backward(torch::Tensor alpha_saved,
                                         torch::Tensor h_saved,
                                         torch::Tensor grad_out);

torch::Tensor scan_forward_half(torch::Tensor alpha, torch::Tensor beta);
std::vector<torch::Tensor> scan_backward_half(torch::Tensor alpha_saved,
                                              torch::Tensor h_saved,
                                              torch::Tensor grad_out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Sequential scan forward and backward pass.",
  m.def("scan_forward", torch::wrap_pybind_function(scan_forward),
        "scan_forward");
  m.def("scan_backward", torch::wrap_pybind_function(scan_backward),
        "scan_backward");
  m.def("scan_forward_half", torch::wrap_pybind_function(scan_forward_half),
        "scan_forward_half");
  m.def("scan_backward_half", torch::wrap_pybind_function(scan_backward_half),
        "scan_backward_half");
}
