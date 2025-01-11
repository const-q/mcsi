from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

setup(
    name="mcsi", # Top-level module name
    ext_modules=[
        CppExtension(
            name="mcsi.mcsi",  # Matches the intended import name
            sources=["mcsi/mcsi.cpp"], # Path to the C++ source file
        )
    ],
    cmdclass={"build_ext": BuildExtension}, # Use PyTorch's build system
    zip_safe=False # Ensures compatibility for extension modules
)
