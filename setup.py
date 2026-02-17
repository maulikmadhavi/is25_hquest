from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import pybind11

class get_pybind_include:
    def __str__(self):
        return pybind11.get_include()

# Determine compile flags based on platform
if sys.platform == 'win32':
    compile_args = ['/O2']
else:
    compile_args = ['-O3', '-Wall', '-shared', '-fPIC']

ext_modules = [
    # Basic DTW module
    Extension(
        'dtw_cpp',
        ['src/dtw_cpp.cpp'],
        include_dirs=[
            get_pybind_include(),
        ],
        language='c++',
        extra_compile_args=compile_args,
    ),
]

setup(
    name='dtw_cpp',
    version='1.0.0',
    author='Your Name',
    description='Fast C++ implementation of Dynamic Time Warping',

    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
