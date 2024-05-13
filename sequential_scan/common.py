import math
from typing import Tuple

import torch

SINGLE_TOL = 7.0
BFLOAT16_TOL = 2.0


def relative_error(x: torch.Tensor, y: torch.Tensor) -> float:
    return (torch.linalg.norm(x - y) / torch.linalg.norm(y)).item()


def check_relative_error(err: float, digits: float):
    if err == 0.0:
        return True
    return math.ceil(-1.0 * math.log10(err)) >= digits


def make_tensors(bs: int, sq: int, d_model: int) -> Tuple[torch.Tensor, torch.Tensor]:
    t1 = torch.randn(
        (bs, sq, d_model), device="cuda:0", dtype=torch.bfloat16, requires_grad=True
    )

    t2 = torch.randn(
        (bs, sq, d_model), device="cuda:0", dtype=torch.bfloat16, requires_grad=True
    )

    return t1, t2


def make_tensors_single(
    bs: int, sq: int, d_model: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    t1 = torch.randn(
        (bs, sq, d_model), device="cuda:0", dtype=torch.float32, requires_grad=True
    )

    t2 = torch.randn(
        (bs, sq, d_model), device="cuda:0", dtype=torch.float32, requires_grad=True
    )

    return t1, t2


def torch_ref(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Reference implementation."""

    out = torch.empty_like(alpha)  # h_t

    out[:, 0, :] = beta[:, 0, :]

    h_rec = beta[:, 0, :]

    for index in range(1, alpha.shape[1]):
        h_rec = (h_rec * alpha[:, index, :]) + beta[:, index, :]
        out[:, index, :] = h_rec

    return out
