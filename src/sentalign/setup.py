from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
    "sentalign.dp_core",
    sources=["src/sentalign/dp_core.pyx"],
    include_dirs=[np.get_include()],
)
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
)
