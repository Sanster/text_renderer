# Text Renderer
Generate text images for training deep learning ocr model.

![example1.jpg](./imgs/example1.jpg)
![example2.jpg](./imgs/example2.jpg)

![example3.jpg](./imgs/example3.jpg)
![example4.jpg](./imgs/example4.jpg)

# Setup
The code was only tested on Ubuntu 16.04.

Install dependencies:
```
pip3 install -r requirements.txt
```

# Generate image
Run `python3 main.py`, images and labels.txt will generate at `output/default/`

Some optional arguments:
- config_file: control the params used when render images
- num_img: how many images to generate
- length: chars or words in a generated image
- output_dir: where to save the images
- corpus_dir: recursively find all txt file in corpus_dir
- corpus_mode: different corpus types have different way to generate samples
    - random: random pick chars from chars file
    - chn: pick continuous chars from corpus
    - eng: pick words from corpus, space is included in label
- chars_file: chars not contained in chars_file will be filtered
- bg_dir: 50% image background are loaded from background image dir

# Strict mode
If some chars in corpus is not supported by your font, your will get bad result:

![bad_example1](./imgs/bad_example1.jpg)

![bad_example2](./imgs/bad_example2.jpg)

![bad_example3](./imgs/bad_example3.jpg)

Run `main.py` with `--strict`, renderer will retry get sample from corpus until all chars are supported by a font.

# Tools
Check how many chars your font not support for a charset:
```bash
python3 tools/check_font.py

checking font ./data/fonts/eng/Hack-Regular.ttf
chars not supported(4971):
['第', '朱', '广', '沪', '联', '自', '治', '县', '驼', '身', '进', '行', '纳', '税', '防', '火', '墙', '掏', '心', '内', '容', '万', '警','钟', '上', '了', '解'...]
0 fonts support all chars(5071) in ./data/chars/chn.txt:
[]
```

# Debug mode
Run `python3 main.py --debug` will save images with extract information.
You can see how perspectiveTransform works and all bounding/rotated boxes.

![debug_demo](./imgs/debug_demo.jpg)

# Generate image using GPU
If you want to use GPU to speed up image generating, first compile opencv with CUDA.
[Compiling OpenCV with CUDA support](https://www.pyimagesearch.com/2016/07/11/compiling-opencv-with-cuda-support/)

Then build Cython part, and add `--gpu` options when run main.py
```
cd libs/gpu
python3 setup.py build_ext --inplace
```


# Todo
See https://github.com/Sanster/text_renderer/projects/1
