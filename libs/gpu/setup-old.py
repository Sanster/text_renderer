import subprocess
import os
import numpy as np
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

"""
Run setup with the following command:
```
python setupGpuWrapper.py build_ext --inplace
```
"""

# Determine current directory of this setup file to find our module
CUR_DIR = os.path.dirname(__file__)
# Use pkg-config to determine library locations and include locations
opencv_libs_str = subprocess.check_output("pkg-config --libs opencv".split()).decode()
opencv_incs_str = subprocess.check_output("pkg-config --cflags opencv".split()).decode()
# Parse into usable format for Extension call
opencv_libs = [str(lib) for lib in opencv_libs_str.strip().split()]
opencv_incs = [str(inc) for inc in opencv_incs_str.strip().split()]

extensions = [
    Extension('GpuWrapper',
              sources=[os.path.join(CUR_DIR, 'GpuWrapper.pyx')],
              language='c++',
              include_dirs=[np.get_include()] + opencv_incs,
              extra_link_args=opencv_libs)
]

setup(
    cmdclass={'build_ext': build_ext},
    name="GpuWrapper",
    ext_modules=cythonize(extensions)
)
