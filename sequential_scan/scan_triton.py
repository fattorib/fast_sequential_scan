from typing import Any, Tuple

import torch
from torch.autograd import Function

from .kernels import sequential_scan_backward, sequential_scan_forward


class SequentialScan(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(
        ctx: Any, input_alpha: torch.Tensor, input_beta: torch.Tensor
    ) -> torch.Tensor:
        h = sequential_scan_forward(input_alpha, input_beta)
        ctx.save_for_backward(input_alpha, h)

        return h

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        (input_alpha, h) = ctx.saved_tensors

        alpha_grad, beta_grad = sequential_scan_backward(input_alpha, h, grad_output)

        return alpha_grad, beta_grad, None


def scan(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return SequentialScan.apply(alpha, beta)
