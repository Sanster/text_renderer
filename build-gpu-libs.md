## Build libs with GPU support

>By Zhuo Zhang  
>2020-07-26 17:37:01  
>https://github.com/zchrissirhcz  


We can build `libs` folder with GPU support, which binds OpenCV 3.x's CUDA API calls with provided glue codes.

### Practice1: Windows 10
For example, OpenCV 3.4.11 with Python 3.7.4(via Miniconda) with NVIDIA GTX 1080Ti on Windows 10:

CUDA version is 11.0, but other versions will be OK.

Visual Studio 2017 version is 15.9.25, other versions not tested.

Assuming Python installed via Anaconda/Miniconda and its version is 3.7.4, other versions not tested.


In git bash, do the following:
```bash
mkdir -p D:/work
cd work

# opencv-3.4.11
cd ~/work;
git clone https://gitee.com/mirros/opencv opencv-3.4.11
cd opencv-3.4.11
git checkout -b 3.4.11 3.4.11

# opencv_contrib-3.4.11
cd ..
git clone https://gitee.com/mirrors/opencv_contrib opencv_contrib-3.4.11
cd opencv_contrib-3.4.11
git checkout -b 3.4.11 3.4.11

# nvidia card compute capability
cd ..
git clone https://github.com/zchrissirhcz/check_ComputeCapability
cd check_ComputeCapability
make
./check_cc

# prepare compile script
cd ../opencv-3.4.11
mkdir -p build
cd build
notepad++ vs2017-x64-gpu.bat
```

`build/vs2017-x64-gpu.bat`'s content:
```batch
@echo off

set BUILD_DIR=vs2017-x64-gpu-cuda11

if not exist %BUILD_DIR% md %BUILD_DIR%

cd %BUILD_DIR%

cmake -G "Visual Studio 15 2017 Win64" ^
    -D CMAKE_BUILD_TYPE=Release ^
    -D CUDA_TOOLKIT_ROOT_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0" ^
    -D CMAKE_INSTALL_PREFIX=D:/lib/opencv/3.4.11 ^
    -D OPENCV_EXTRA_MODULES_PATH=D:/work/opencv_contrib-3.4.11/modules ^
    -D WITH_CUDA=ON ^
    -D CUDA_ARCH_BIN=6.1 ^
    -D WITH_VTK=OFF ^
    -D WITH_MATLAB=OFF ^
    -D BUILD_DOCS=OFF ^
    -D BUILD_opencv_python3=ON ^
    -D BUILD_opencv_python2=OFF ^
    -D PYTHON3_EXECUTABLE=D:/soft/Miniconda3/python.exe ^
    -D PYTHON_INCLUDE_DIR=D:/soft/Miniconda3/include ^
    -D PYTHON_LIBRARY=D:/soft/Miniconda3/libs/python37.lib ^
    -D WITH_FFMPEG=OFF ^
    -D BUILD_JAVA=OFF ^
    -D WITH_PROTOBUF=OFF ^
    -D WITH_IPP=OFF ^
    -D BUILD_TESTS=OFF ^
    -D BUILD_PERF_TESTS=OFF ^
    -D WITH_OPENCL=OFF ^
    ../..

cd ..

pause
```

Then, open a cmd, do these:
```batch
D:
cd D:/work/opencv-3.4.11/build
run vs2017-x64-gpu.bat
```

Then open `D:/work/opencv-3.4.11/build/vs2017-x64-gpu/opencv.sln` with Visual Studio 2017, switch to **Release** mode, choose **INSTALL** target and build it.

Later, open another cmd, do the following:
```bash
# get text_renderer's code
D:
cd D:/work
git clone https://github.com/zchrissirhcz/text_renderer
cd text_renderder
cd libs
cd gpu

# Let's make sure VS2017 will be called, instead of other version of VS if there's multiple
call "C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/vcvarsall.bat" amd64
python setup.py build_ext --inplace
```

**Note: if you're using different opencv directory, please modify it in `setup.py` (search `write_cmakelist` as keyword) before building.**

Finally, use the compiled OpenCV Python lib and GpuWrapper lib together, i.e. in `math_utils.py`, modify the imports to be like this:
```Python
import sys
sys.path.insert(1, 'D:/work/opencv-3.4.11/build/vs2017-x64-gpu-cuda11/python_loader')
import cv2
import GpuWrapper
```

And add `--gpu` option when run `main.py`


### Practice2: Ubuntu
For example, OpenCV 3.4.11 with Python 3.8.1(via Miniconda) with NVIDIA GTX 1080Ti on Ubuntu 20.04:

```bash
# opencv-3.4.11
cd ~/work;
git clone https://gitee.com/mirros/opencv opencv-3.4.11
cd opencv-3.4.11
git checkout -b 3.4.11 3.4.11

# opencv_contrib-3.4.11
cd ..
git clone https://gitee.com/mirrors/opencv_contrib opencv_contrib-3.4.11
cd opencv_contrib-3.4.11
git checkout -b 3.4.11 3.4.11

# nvidia card compute capability
cd ..
git clone https://github.com/zchrissirhcz/check_ComputeCapability
cd check_ComputeCapability
make
./check_cc

# prepare compile script
cd ../opencv-3.4.11
vim compile.sh
```

`compile.sh`'s content:
```bash
#!/bin/bash

set -x
set -e

mkdir -p $HOME/soft

rm -rf build
mkdir -p build
cd build

LOG="../cmake.log"
touch $LOG
rm $LOG

exec &> >(tee -a "$LOG")

cmake .. \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=$HOME/soft/opencv-3.4.11 \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=6.1 \
    -D OPENCV_EXTRA_MODULES_PATH=$HOME/work/opencv_contrib-3.4.11/modules \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_PYTHON3_VERSION=3.8 \
    -D PYTHON3_EXECUTABLE=$HOME/soft/miniconda3/bin/python \
    -D PYTHON3_INCLUDE_DIR=$HOME/soft/miniconda3/include/python3.8 \
    -D PYTHON3_LIBRARY=$HOME/soft/miniconda3/lib/libpython3.8.so \
    -D BUILD_opencv_python3=ON \
    -D BUILD_opencv_python2=OFF \
    -D PYTHON_DEFAULT_EXECUTABLE=$HOME/soft/miniconda3/bin/python \
    -D HAVE_opencv_python3=ON \
    -D BUILD_TIFF=ON \
    -D WITH_VTK=OFF \
    -D WITH_MATLAB=OFF \
    -D BUILD_DOCS=OFF \

make -j4
make install
```

Then do:
```bash
# compile opencv
./compile.sh

# compile cython module
cd libs/gpu
python setup.py build_ext --inplace
```

**Note: if you're using different opencv directory, please modify it in `setup.py` (search `write_cmakelist` as keyword) before building.**

Then use this built opencv2 python library and GpuWrapper together, such as in `math_utils.py`. Modify its imports like in the windows example.

Finally, and add `--gpu` option when run `main.py`

### References
- [Accessing OpenCV CUDA Functions from Python (No PyCUDA)](https://stackoverflow.com/questions/42125084/accessing-opencv-cuda-functions-from-python-no-pycuda)

- [Compiling OpenCV with CUDA support](https://www.pyimagesearch.com/2016/07/11/compiling-opencv-with-cuda-support/)

