from typing import Any, Tuple

import torch
from sequential_scan_cuda import (
    scan_backward,
    scan_backward_half,
    scan_forward,
    scan_forward_half,
)
from torch.autograd import Function


class SequentialScan(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(
        ctx: Any, input_alpha: torch.HalfTensor, input_beta: torch.HalfTensor
    ) -> torch.HalfTensor:
        h = scan_forward_half(input_alpha, input_beta)
        ctx.save_for_backward(input_alpha, h)

        return h

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(
        ctx: Any, grad_output: torch.HalfTensor
    ) -> Tuple[torch.HalfTensor, torch.HalfTensor, None]:
        (input_alpha, h) = ctx.saved_tensors

        alpha_grad, beta_grad = scan_backward_half(input_alpha, h, grad_output)

        return alpha_grad, beta_grad, None


class SequentialScanSingle(Function):
    @staticmethod
    def forward(
        ctx: Any, input_alpha: torch.Tensor, input_beta: torch.Tensor
    ) -> torch.Tensor:
        h = scan_forward(input_alpha, input_beta)
        ctx.save_for_backward(input_alpha, h)

        return h

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        (input_alpha, h) = ctx.saved_tensors

        alpha_grad, beta_grad = scan_backward(input_alpha, h, grad_output)

        return alpha_grad, beta_grad, None


def scan(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return SequentialScan.apply(alpha, beta)


def scan_single(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return SequentialScanSingle.apply(alpha, beta)
