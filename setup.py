from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("compression.utils_quick", ["compression/utils_quick.pyx"],
              include_dirs=[np.get_include(), '.']),
]

setup(
    ext_modules=cythonize(extensions)
)
