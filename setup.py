from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="sequential_scan",
    version="1.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            "sequential_scan_cuda",
            [
                "csrc/sequential_scan.cpp",
                "csrc/sequential_scan_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-arch=compute_89",
                    "-I csrc/include",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
