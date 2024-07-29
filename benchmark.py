import argparse

import torch
import triton
from accelerated_scan.warp import scan as scan_acc  # pip install accelerated-scan

from sequential_scan import scan_cu, scan_cu_single, scan_tr


def create_tensor(bs: int, sq: int, d_model: int) -> torch.Tensor:
    """Problem setup."""
    return torch.randn(
        (bs, sq, d_model), device="cuda:0", dtype=torch.bfloat16, requires_grad=True
    )
    # return torch.randn(
    #     (bs, sq, d_model), device="cuda:0", dtype=torch.float32, requires_grad=True
    # )


def parse():
    parser = argparse.ArgumentParser(description="Run the scan benchmark.")
    parser.add_argument("--d_model", type=int, required=True, help="Model dimension")
    parser.add_argument("--bs", type=int, required=True, help="Batch size")
    parser.add_argument(
        "--plot_name", type=str, required=True, help="Name for the plot"
    )
    parser.add_argument(
        "--fwd_only",
        action="store_false",
        help="Runs forward and backward passes for benchmark.",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse()

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["sq"],
            x_vals=[128, 256, 1024, 2048, 4096, 8192, 16384, 32768],
            line_arg="provider",
            line_vals=["accelerated-scan", "triton", "cuda-bf16"],
            line_names=[
                "accelerated-scan",
                "Triton Kernel",
                "CUDA Kernel",
            ],
            styles=[("blue", "-"), ("green", "-"), ("red", "-"), ("orange", "-")],
            ylabel="time (ms)",
            plot_name=args.plot_name,
            args={"d_model": args.d_model, "bs": args.bs, "run_bwd": args.fwd_only},
        )
    )
    def bench_scan(bs, sq, d_model, provider, run_bwd):
        # create data

        alpha = create_tensor(bs, sq, d_model=d_model)
        beta = create_tensor(bs, sq, d_model=d_model)

        dy = 0.1 * torch.randn_like(alpha)

        quantiles = [0.5, 0.2, 0.8]
        # utility functions

        if provider == "accelerated-scan":
            # tensor order is different here
            a = alpha.permute(0, 2, 1).contiguous()
            b = beta.permute(0, 2, 1).contiguous()
            d = dy.permute(0, 2, 1).contiguous()

            def y_fwd():
                return scan_acc(a, b)

            def y_fwd_bwd():
                y = y_fwd()
                return y.backward(d)

        if provider == "triton":

            def y_fwd():
                return scan_tr(alpha, beta)

            def y_fwd_bwd():
                y = y_fwd()
                return y.backward(dy)

        if provider == "cuda-bf16":

            def y_fwd():
                return scan_cu(alpha, beta)
                # return scan_cu_single(alpha, beta)

            def y_fwd_bwd():
                y = y_fwd()
                return y.backward(dy)

        if run_bwd:
            fn = y_fwd_bwd

        else:
            fn = y_fwd

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, rep=100)

        return ms

    bench_scan.run(save_path=".", print_data=True)


if __name__ == "__main__":
    main()
