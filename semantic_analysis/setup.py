from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("feature_extraction.pyx", "gspan_mining/graph.pyx", "gspan_mining/gspan.pyx")
)