import os, sys
import os.path as osp
from glob import glob
from setuptools import setup, find_packages

import torch
from torch.__config__ import parallel_info
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDAExtension

# check machine exists gpu
if not torch.cuda.is_available():
    raise Exception("Machine must have cuda device!")

# check cuda samples include dir
cuda_samples_include_dir = os.getenv("CUDA_SAMPLES_INCLUDE_DIR", "/usr/local/cuda/samples/common/inc")
if not osp.exists(cuda_samples_include_dir):
    raise Exception(f"Cuda samples include dir: {cuda_samples_include_dir} doesn't exist! You may specify environment variable `CUDA_SAMPLES_INCLUDE_DIR` to resolve this problem.")

# must compile with openmp
info = parallel_info()
if ("backend: OpenMP" not in info) or ("OpenMP not found" in info):
    raise Exception("Must compile with openmp!")


def get_extension():
    extra_compile_args = dict(cxx=["-O3", 
                                   "-DAT_PARALLEL_OPENMP", 
                                   "-fopenmp" if sys.platform == "win32" else "/openmp",
                                   ],
                              nvcc=["-O3",
                                    "-w", 
                                    "-Xptxas=-w", 
                                    "-Xcompiler=-fopenmp,-funroll-loops",
                                    ],
                            )

    return CUDAExtension(
        name="torch_kdtree",
        sources=glob("src/*.cu"),
        include_dirs=["src", cuda_samples_include_dir],
        define_macros=[("TORCH_EXTENSION_NAME", "torch_kdtree")],
        extra_compile_args=extra_compile_args,
        extra_link_args=[],
    )


setup(
    name="torch_kdtree",
    version="0.0",
    author="Jiabao Lei",
    author_email="eejblei@mail.scut.edu.cn",
    url="https://github.com/Karbo123/torch_kdtree",
    description="A CUDA implementation of KDTree in PyTorch",
    keywords=["pytorch", "kdtree", "cuda", "search"],
    license="MIT",
    python_requires=">=3.7",
    install_requires=[],
    setup_requires=[],
    tests_require=[],
    extras_require=dict(),
    ext_modules=[get_extension()],
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    },
    packages=find_packages(),
)
