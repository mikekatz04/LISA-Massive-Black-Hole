# from future.utils import iteritems
import os
import sys
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import shutil
import argparse

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# build all cpu modules
cpu_extension = dict()

HetLike_ext = Extension(
    "pyHetWrap",
    sources=[
        "Utils.cc",
        "IMRPhenomD_internals.cc",
        "IMRPhenomD.cc",
        "Response.cc",
        "het_wrap.pyx",
    ],
    libraries=["gsl", "gslcblas"],
    language="c++",
    runtime_library_dirs=[],
    extra_compile_args=["-w"],  # "-std=c++11"],  # '-g'
    include_dirs=[numpy_include, "."],
    library_dirs=None,
)

setup(
    name="hetwrap",
    ext_modules=[HetLike_ext],
    python_requires=">=3.6",
)
