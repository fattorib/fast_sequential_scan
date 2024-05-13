import numpy as np
import pytest
import torch

from .common import (
    check_relative_error,
    make_tensors,
    make_tensors_single,
    relative_error,
    torch_ref,
)
from .scan_cuda import scan, scan_single

torch.manual_seed(0)
torch.manual_seed(0)


SINGLE_TOL = 7.0
BFLOAT16_TOL = 2.0


# --------------------------------
# Single Precision (torch.float32)
# --------------------------------


@pytest.mark.parametrize("bs", [9, 10, 12, 2])
@pytest.mark.parametrize("sq", [128, 256, 512, 1024])
@pytest.mark.parametrize("d", [512, 768, 1024])
def test_fwd_single(bs, sq, d):

    alpha, beta = make_tensors_single(bs, sq, d)

    with torch.no_grad():
        out_torch = torch_ref(alpha, beta)

    with torch.no_grad():
        out_cuda = scan_single(alpha, beta)

    assert check_relative_error(
        relative_error(out_cuda, out_torch), SINGLE_TOL
    ), relative_error(out_cuda, out_torch)


@pytest.mark.parametrize("bs", [9, 10, 12, 2])
@pytest.mark.parametrize("sq", [128, 256, 512, 1024])
@pytest.mark.parametrize("d", [512, 768, 1024])
def test_bwd_single(bs, sq, d):

    alpha, beta = make_tensors_single(bs, sq, d)

    alpha.requires_grad_(True), beta.requires_grad_(True)

    out_torch = torch_ref(alpha, beta)

    out_cuda = scan_single(alpha, beta)

    dy = torch.randn_like(out_torch)

    out_torch.backward(dy, retain_graph=True)

    dalpha_torch, dbeta_torch = [_.grad.clone() for _ in [alpha, beta]]
    alpha.grad, beta.grad = None, None

    out_cuda.backward(dy, retain_graph=True)
    dalpha_cuda, dbeta_cuda = [_.grad.clone() for _ in [alpha, beta]]

    assert check_relative_error(
        relative_error(dalpha_cuda, dalpha_torch), SINGLE_TOL
    ), relative_error(dalpha_cuda, dalpha_torch)

    assert check_relative_error(
        relative_error(dbeta_cuda, dbeta_torch), SINGLE_TOL
    ), relative_error(dbeta_cuda, dbeta_torch)


# -------------------------------
# Half Precision (torch.bfloat16)
# -------------------------------


@pytest.mark.parametrize("bs", np.random.randint(low=1, high=16, size=5))
@pytest.mark.parametrize("sq", [128, 256, 512, 1024])
@pytest.mark.parametrize("d", [512, 768, 1024])
def test_fwd_half(bs, sq, d):

    alpha, beta = make_tensors(bs, sq, d)

    with torch.no_grad():
        out_torch = torch_ref(alpha, beta)

    with torch.no_grad():
        out_cuda = scan(alpha, beta)

    assert check_relative_error(
        relative_error(out_cuda, out_torch), BFLOAT16_TOL
    ), relative_error(out_cuda, out_torch)


@pytest.mark.parametrize("bs", np.random.randint(low=1, high=16, size=5))
@pytest.mark.parametrize("sq", [128, 256, 512, 1024])
@pytest.mark.parametrize("d", [512, 768, 1024])
def test_bwd_half(bs, sq, d):

    alpha, beta = make_tensors(bs, sq, d)
    alpha.requires_grad_(True), beta.requires_grad_(True)

    out_torch = torch_ref(alpha, beta)

    out_cuda = scan(alpha, beta)

    dy = torch.randn_like(out_torch)

    out_torch.backward(dy, retain_graph=True)

    dalpha_torch, dbeta_torch = [_.grad.clone() for _ in [alpha, beta]]
    alpha.grad, beta.grad = None, None

    out_cuda.backward(dy, retain_graph=True)
    dalpha_cuda, dbeta_cuda = [_.grad.clone() for _ in [alpha, beta]]

    assert check_relative_error(
        relative_error(dalpha_cuda, dalpha_torch), BFLOAT16_TOL
    ), relative_error(dalpha_cuda, dalpha_torch)

    assert check_relative_error(
        relative_error(dbeta_cuda, dbeta_torch), BFLOAT16_TOL
    ), relative_error(dbeta_cuda, dbeta_torch)
