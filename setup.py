from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fla_volta",
    version="0.1.0",
    description="FLA CUDA kernels for Volta (sm_70) — replaces Triton ops that fail on V100",
    author="INMECHA INC",
    packages=["fla_volta"],
    ext_modules=[
        CUDAExtension(
            name="fla_volta._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/fused_norm_gate.cu",
                "csrc/gated_delta_net.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-arch=sm_70",
                    "--use_fast_math",
                    "-lineinfo",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
