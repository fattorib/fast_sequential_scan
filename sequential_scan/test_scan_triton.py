import numpy as np
import pytest
import torch

from .common import check_relative_error, make_tensors, relative_error, torch_ref
from .scan_triton import scan

torch.manual_seed(0)
np.random.seed(0)

BFLOAT16_TOL = 2


@pytest.mark.parametrize("bs", np.random.randint(low=1, high=16, size=5))
@pytest.mark.parametrize("sq", [128, 256, 512, 1024])
@pytest.mark.parametrize("d", [64, 128, 384, 768])
def test_fwd(bs, sq, d):
    alpha, beta = make_tensors(bs, sq, d)

    with torch.no_grad():
        out_torch = torch_ref(alpha, beta)

    with torch.no_grad():
        out_triton = scan(alpha, beta)

    assert check_relative_error(
        relative_error(out_triton, out_torch), BFLOAT16_TOL
    ), relative_error(out_triton, out_torch)


@pytest.mark.parametrize("bs", np.random.randint(low=1, high=16, size=5))
@pytest.mark.parametrize("sq", [128, 256, 512, 1024])
@pytest.mark.parametrize("d", [64, 128, 384, 768])
def test_bwd(bs, sq, d):
    alpha, beta = make_tensors(bs, sq, d)

    out_torch = torch_ref(alpha, beta)

    out_triton = scan(alpha, beta)

    dy = torch.randn_like(out_torch)

    out_torch.backward(dy, retain_graph=True)

    dalpha_torch, dbeta_torch = [_.grad.clone() for _ in [alpha, beta]]
    alpha.grad, beta.grad = None, None

    out_triton.backward(dy, retain_graph=True)
    dalpha_triton, dbeta_triton = [_.grad.clone() for _ in [alpha, beta]]

    assert check_relative_error(
        relative_error(dalpha_triton, dalpha_torch), BFLOAT16_TOL
    ), relative_error(dalpha_triton, dalpha_torch)

    assert check_relative_error(
        relative_error(dbeta_triton, dbeta_torch), BFLOAT16_TOL
    ), relative_error(dbeta_triton, dbeta_torch)
