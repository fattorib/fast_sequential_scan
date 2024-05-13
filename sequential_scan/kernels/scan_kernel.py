from typing import Tuple

import torch
import triton
import triton.language as tl


# fmt: off
@triton.jit
def sequential_scan_fwd_kernel(
    alpha_ptr,beta_ptr,hidden_ptr,
    bs_stride,sq_stride,
    num_context: tl.constexpr,
    numel: tl.constexpr,
    BLOCKSIZE: tl.constexpr
):
    #fmt: on

    bs_pid = tl.program_id(0)

    alpha_ptr += bs_pid * bs_stride
    beta_ptr += bs_pid * bs_stride
    hidden_ptr += bs_pid * bs_stride

    offs = tl.arange(0, BLOCKSIZE)
    mask = offs < numel
    # compute h_0 outside loop

    hidden_t = tl.load(beta_ptr + offs, mask = mask)

    tl.store(hidden_ptr + offs, hidden_t, mask = mask)

    for i in range(1, num_context):
        beta_ptr += sq_stride
        alpha_ptr += sq_stride
        hidden_ptr += sq_stride

        alpha_t = tl.load(alpha_ptr + offs, mask = mask)
        beta_t = tl.load(beta_ptr + offs, mask = mask)

        hidden_t = alpha_t * hidden_t + beta_t

        tl.store(hidden_ptr + offs, hidden_t, mask = mask)

#fmt: off
@triton.jit
def sequential_scan_bwd_kernel(
    alpha_saved_ptr,h_saved_ptr,d_out_ptr,
    d_alpha_ptr,d_beta_ptr, 
    bs_stride, sq_stride,
    num_context: tl.constexpr,
    numel: tl.constexpr,
    BLOCKSIZE: tl.constexpr
):
    #fmt: on
    bs_pid = tl.program_id(0)
    

    # offset ptrs to correct batch start 
    alpha_saved_ptr += (bs_pid * bs_stride) + ((num_context)*sq_stride)
    h_saved_ptr += (bs_pid * bs_stride) + ((num_context -2)*sq_stride) 
    d_out_ptr += (bs_pid * bs_stride) + ((num_context-1)*sq_stride)

    d_alpha_ptr += (bs_pid * bs_stride) + ((num_context-1)*sq_stride)
    d_beta_ptr += (bs_pid * bs_stride) + ((num_context-1)*sq_stride)

    offs = tl.arange(0, BLOCKSIZE)
    
    mask = offs < numel

    # compute (t = T) outside loop
    h_grad = tl.load(d_out_ptr + offs, mask=mask)
    h_rec = tl.load(h_saved_ptr + offs, mask=mask)

    d_alpha = h_grad*h_rec
    d_beta = h_grad

    tl.store(d_alpha_ptr + offs, d_alpha,mask=mask)
    tl.store(d_beta_ptr + offs, d_beta,mask=mask)
    
    for _ in range(2, num_context):
        # reduce pointer offsets
        d_alpha_ptr -= sq_stride
        d_beta_ptr -= sq_stride
        h_saved_ptr -= sq_stride
        d_out_ptr -= sq_stride
        alpha_saved_ptr -= sq_stride

        alpha = tl.load(alpha_saved_ptr + offs,mask=mask)
        grad_out = tl.load(d_out_ptr + offs,mask=mask)
        h_rec = tl.load(h_saved_ptr + offs,mask=mask)


        h_grad = alpha * h_grad
        h_grad += grad_out
        
        d_alpha = h_grad * h_rec
        d_beta = h_grad

        tl.store(d_alpha_ptr + offs, d_alpha,mask=mask)
        tl.store(d_beta_ptr + offs, d_beta,mask=mask)


    # first grad (t = 0)
    d_alpha_ptr -= sq_stride
    d_beta_ptr -= sq_stride
    d_out_ptr -= sq_stride
    alpha_saved_ptr -= sq_stride

    alpha = tl.load(alpha_saved_ptr + offs,mask=mask)
    grad_out = tl.load(d_out_ptr + offs,mask=mask)
    
    h_grad = alpha * h_grad
    h_grad += grad_out
    d_beta = h_grad

    d_alpha = tl.zeros_like(d_beta)

    tl.store(d_alpha_ptr + offs, d_alpha,mask=mask)
    tl.store(d_beta_ptr + offs, d_beta,mask=mask)

def sequential_scan_forward(
    alpha: torch.Tensor,  # [b,sq,d]
    beta: torch.Tensor,  # [b,sq,d]
) -> torch.Tensor:
    """Computes forward pass of a linear scan."""

    hidden = torch.empty_like(beta)

    b, sq, d = alpha.shape

    BLOCKSIZE = triton.next_power_of_2(d)

    grid = (b,)

    warps = 4 if d <= 1024 else 8

    #fmt: off
    sequential_scan_fwd_kernel[grid](
        alpha,beta,hidden,
        alpha.stride(0),alpha.stride(1),
        sq,d,BLOCKSIZE,
        num_warps = warps
    )
    #fmt: on
    return hidden


def sequential_scan_backward(
    alpha_saved: torch.Tensor,  # [b,sq,d]
    h_saved: torch.Tensor,  # [b,sq,d]
    grad_out: torch.Tensor,  # [b,sq,d]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes backward pass of a linear scan."""

    alpha_grad = torch.empty_like(alpha_saved)
    beta_grad = h_saved
    
    b, sq, d = alpha_saved.shape

    BLOCKSIZE = triton.next_power_of_2(d)

    grid = (b,)

    warps = 4 if d <= 1024 else 8

    #fmt: off
    sequential_scan_bwd_kernel[grid](
        alpha_saved,h_saved, grad_out,
        alpha_grad, beta_grad,
        alpha_saved.stride(0), alpha_saved.stride(1),
        sq, d, BLOCKSIZE, num_warps = warps
    )
    #fmt: on

    return alpha_grad, beta_grad
