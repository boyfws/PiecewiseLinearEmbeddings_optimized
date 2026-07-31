from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            name="rtdl_num_embeddings_cuda._C",
            sources=[
                "csrc/ops.cpp",
                "csrc/forward_cuda.cu",
                "csrc/backward_cuda.cu",
            ],
            include_dirs=[
                "csrc",
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++17",
                ],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-lineinfo",
                ],
            },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=True,
        ),
    },
    zip_safe=False,
)
