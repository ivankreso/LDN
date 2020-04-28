from distutils.core import setup
from Cython.Build import cythonize

# python setup.py build_ext --inplace

# setup(
#   ext_modules = cythonize("cylib.pyx")
# )

setup(ext_modules = cythonize(
          "cylib.pyx",                 # our Cython source
          language="c++",             # generate C++ code
))
