from __future__ import annotations

import os

from setuptools import find_packages, setup

ext_modules = []
cmdclass = {}

if os.getenv("HELIX_BUILD_CUDA_EXT", "0") == "1":
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        ext_modules.append(
            CUDAExtension(
                name="helix_cuda_ext",
                sources=[
                    "cuda_ext/csrc/cu_seqlens.cpp",
                    "cuda_ext/csrc/cu_seqlens_kernel.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": ["-O3", "--use_fast_math"],
                },
            )
        )
        cmdclass["build_ext"] = BuildExtension
    except Exception as exc:
        print(f"[setup.py] Skipping CUDA extension build: {exc}")

setup(
    name="helixserve",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
