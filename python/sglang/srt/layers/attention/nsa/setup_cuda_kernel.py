"""Setup script for compiling CUDA kernel"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the directory of this file
current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='transform_index_cuda',
    ext_modules=[
        CUDAExtension(
            name='transform_index_cuda',
            sources=[os.path.join(current_dir, 'transform_index_cuda.cu')],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',
                    '--expt-relaxed-constexpr',
                    '-gencode', 'arch=compute_80,code=sm_80',  # A100
                    '-gencode', 'arch=compute_86,code=sm_86',  # RTX 3090
                    '-gencode', 'arch=compute_89,code=sm_89',  # RTX 4090
                    '-gencode', 'arch=compute_90,code=sm_90',  # H100
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

