import subprocess
import os
import numpy as np
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

from distutils.spawn import find_executable
import sys
from distutils import sysconfig, log
from contextlib import contextmanager

"""
Run setup with the following command:
```
python setupGpuWrapper.py build_ext --inplace
```
"""

@contextmanager
def cd(path):
    if not os.path.isabs(path):
        raise RuntimeError('Can only cd to absolute path, got: {}'.format(path))
    if os.path.isdir(path) is False:
        os.makedirs(path)
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


def write_cmakelist(opencv_install_dir):
    """
    @param opencv_install_dir: the directory that contains 'OpenCVConfig.cmake'
    """
    lines = []
    lines.append('cmake_minimum_required(VERSION 3.6)')
    lines.append('project(qq)')
    lines.append('set(CMAKE_CXX_STANDARD 11)')
    lines.append('set(OpenCV_DIR "{:s}" CACHE PATH "")'.format(opencv_install_dir))
    lines.append('find_package(OpenCV)')
    lines.append('set(MYDEP_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} CACHE PATH "")')
    lines.append('set(MYDEP_LIBS ${OpenCV_LIBS} CACHE PATH "")')
    lines.append('set(MYDEP_LIBRARY_DIRS "${OpenCV_DIR}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/lib" CACHE PATH "")')

    fout = open('CMakeLists.txt', 'w', encoding='utf-8')
    for line in lines:
        fout.write(line+"\n")
    fout.close()


cmake = find_executable('cmake')
assert cmake, 'Could not find "cmake" executable!'

# You may modify me here!
write_cmakelist("D:/work/opencv-4.3.0/build/vs2017-x64-gpu/install")

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
CMAKE_BUILD_DIR = os.path.join(TOP_DIR, '.setuptools-cmake-build')

with cd(CMAKE_BUILD_DIR):
    cmake_args = [
        cmake,
        '-DBUILD_SHARED_LIBS=OFF',
        '-DPYTHON_EXECUTABLE:FILEPATH={}'.format(sys.executable),
        '-DPYTHON_INCLUDE_DIR={}'.format(sysconfig.get_python_inc()),
        '-DBUILD_TEST=OFF',
        '-DBUILD_BENCHMARK=OFF',
        '-DBUILD_BINARY=OFF',
        '-G', 'Visual Studio 15 2017 Win64'
    ]
    cmake_args.append(TOP_DIR)
    subprocess.check_call(cmake_args)

cmake_cache_file = os.path.join(CMAKE_BUILD_DIR, 'CMakeCache.txt')

"""
for example:

MYDEP_INCLUDE_DIRS:PATH=D:/work/opencv-4.3.0/build/vs2017-x64-gpu/install/include

MYDEP_LIBRARY_DIRS:PATH=D:/work/opencv-4.3.0/build/vs2017-x64-gpu/install/x64/vc15/lib

MYDEP_LIBS:PATH=opencv_calib3d;opencv_core;opencv_features2d;opencv_flann;opencv_gapi;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_stitching;opencv_video;opencv_videoio;opencv_aruco;opencv_bgsegm;opencv_bioinspired;opencv_ccalib;opencv_cudaarithm;opencv_cudabgsegm;opencv_cudacodec;opencv_cudafeatures2d;opencv_cudafilters;opencv_cudaimgproc;opencv_cudalegacy;opencv_cudaobjdetect;opencv_cudaoptflow;opencv_cudastereo;opencv_cudawarping;opencv_cudev;opencv_datasets;opencv_dpm;opencv_face;opencv_fuzzy;opencv_hfs;opencv_img_hash;opencv_intensity_transform;opencv_line_descriptor;opencv_optflow;opencv_phase_unwrapping;opencv_plot;opencv_quality;opencv_rapid;opencv_reg;opencv_rgbd;opencv_saliency;opencv_shape;opencv_stereo;opencv_structured_light;opencv_superres;opencv_surface_matching;opencv_tracking;opencv_videostab;opencv_xfeatures2d;opencv_ximgproc;opencv_xobjdetect;opencv_xphoto
"""

fin = open(cmake_cache_file, 'r')
for line in fin.readlines():
    line = line.strip()
    if line.startswith('MYDEP_INCLUDE_DIRS'):
        mydep_include_dirs = [line.split('=')[1]]
    if line.startswith('MYDEP_LIBRARY_DIRS'):
        mydep_library_dirs = [line.split('=')[1]]
    if line.startswith('MYDEP_LIBS'):
        mydep_libs = [line.split('=')[1]]
fin.close()



# Determine current directory of this setup file to find our module
CUR_DIR = os.path.dirname(__file__)
# Use pkg-config to determine library locations and include locations
#opencv_libs_str = subprocess.check_output("pkg-config --libs opencv".split()).decode()
#opencv_incs_str = subprocess.check_output("pkg-config --cflags opencv".split()).decode()

# Parse into usable format for Extension call
#opencv_libs = [str(lib) for lib in opencv_libs_str.strip().split()]
#opencv_incs = [str(inc) for inc in opencv_incs_str.strip().split()]

extensions = [
    Extension('GpuWrapper',
              sources=[os.path.join(CUR_DIR, 'GpuWrapper.pyx')],
              language='c++',
              include_dirs=[np.get_include()] + mydep_include_dirs,
              libraries=mydep_libs,
              library_dirs=mydep_library_dirs,
              #define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
              )
              #extra_link_args=opencv_libs)
]

setup(
    cmdclass={'build_ext': build_ext},
    name="GpuWrapper",
    ext_modules=cythonize(extensions)
)
