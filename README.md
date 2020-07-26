**New version release**：https://github.com/oh-my-ocr/text_renderer

# Text Renderer
Generate text images for training deep learning OCR model (e.g. [CRNN](https://github.com/bgshih/crnn)).
Support both latin and non-latin text.

# Setup
- Ubuntu 16.04
- python 3.5+

Install dependencies:
```
pip3 install -r requirements.txt
```

# Demo
By default, simply run `python3 main.py` will generate 20 text images
and a labels.txt file in `output/default/`.

![example1.jpg](./imgs/example1.jpg)
![example2.jpg](./imgs/example2.jpg)

![example3.jpg](./imgs/example3.jpg)
![example4.jpg](./imgs/example4.jpg)

# Use your own data to generate image
1. Please run `python3 main.py --help` to see all optional arguments and their meanings.
And put your own data in corresponding folder.

2. Config text effects and fraction in `configs/default.yaml` file(or create a
new config file and use it by `--config_file` option), here are some examples:

|Effect name|Image|
|------------|----|
|Origin(Font size 25)|![origin](./imgs/effects/origin.jpg)|
|Perspective Transform|![perspective](./imgs/effects/perspective_transform.jpg)|
|Random Crop|![rand_crop](./imgs/effects/random_crop.jpg)|
|Curve|![curve](./imgs/effects/curve.jpg)|
|Light border|![light border](./imgs/effects/light_border.jpg)|
|Dark border|![dark border](./imgs/effects/dark_border.jpg)|
|Random char space big|![random char space big](./imgs/effects/random_space_big.jpg)|
|Random char space small|![random char space small](./imgs/effects/random_space_small.jpg)|
|Middle line|![middle line](./imgs/effects/line_middle.jpg)|
|Table line|![table line](./imgs/effects/line_table.jpg)|
|Under line|![under line](./imgs/effects/line_under.jpg)|
|Emboss|![emboss](./imgs/effects/emboss.jpg)|
|Reverse color|![reverse color](./imgs/effects/reverse.jpg)|
|Blur|![blur](./imgs/effects/blur.jpg)|
|Text color|![font_color](./imgs/effects/colored.jpg)|
|Line color|![line_color](./imgs/effects/table.jpg)|

3. Run `main.py` file.

# Strict mode
For no-latin language(e.g Chinese), it's very common that some fonts only support
limited chars. In this case, you will get bad results like these:

![bad_example1](./imgs/bad_example1.jpg)

![bad_example2](./imgs/bad_example2.jpg)

![bad_example3](./imgs/bad_example3.jpg)

Select fonts that support all chars in `--chars_file` is annoying.
Run `main.py` with `--strict` option, renderer will retry get text from
corpus during generate processing until all chars are supported by a font.

# Tools
You can use `check_font.py` script to check how many chars your font not support in `--chars_file`:
```bash
python3 tools/check_font.py

checking font ./data/fonts/eng/Hack-Regular.ttf
chars not supported(4971):
['第', '朱', '广', '沪', '联', '自', '治', '县', '驼', '身', '进', '行', '纳', '税', '防', '火', '墙', '掏', '心', '内', '容', '万', '警','钟', '上', '了', '解'...]
0 fonts support all chars(5071) in ./data/chars/chn.txt:
[]
```

# Generate image using GPU
If you want to use GPU to make generate image faster, first compile OpenCV 3.x with CUDA (OpenCV 4.x won't work) and Python suport.

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

Finally, and add `--gpu` option when run `main.py`

refs: 
- [Accessing OpenCV CUDA Functions from Python (No PyCUDA)](https://stackoverflow.com/questions/42125084/accessing-opencv-cuda-functions-from-python-no-pycuda)
- [Compiling OpenCV with CUDA support](https://www.pyimagesearch.com/2016/07/11/compiling-opencv-with-cuda-support/)


# Debug mode
Run `python3 main.py --debug` will save images with extract information.
You can see how perspectiveTransform works and all bounding/rotated boxes.

![debug_demo](./imgs/debug_demo.jpg)


# Todo
See https://github.com/Sanster/text_renderer/projects/1
