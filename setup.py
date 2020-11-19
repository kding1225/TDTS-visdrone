#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# compile command: python setup.py build develop --no-deps

import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension


requirements = [
    "torchvision",
    "ninja",
    "yacs",
    "cython",
    "matplotlib",
    "tqdm",
    "opencv-python",
    "scikit-image",
    "spconv"
]


def get_extensions():
    extensions_dir = os.path.join("fcos_core", "csrc")
    
    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
    sources = main_file + source_cpu

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
	        # "-std=c++14"
        ]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "fcos_core._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args
        )
    ]

    return ext_modules


setup(
    name="TDTS",
    version="0.1.0",
    author="Kun Ding",
    url="https://github.com/kding1225/TDTS-VisDrone",
    description="TDTS object detector in pytorch",
    scripts=["fcos/bin/fcos"],
    packages=find_packages(exclude=("configs", "tests",)),
    install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    include_package_data=True,
)
