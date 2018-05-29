# Text Renderer
Generate text images for training deep learning ocr model.

![example1.jpg](./imgs/example1.jpg)
![example2.jpg](./imgs/example2.jpg)

![example3.jpg](./imgs/example3.jpg)
![example4.jpg](./imgs/example4.jpg)

# Setup
Install dependencies:
```
pip3 install -r requirements.txt
```

# Generate image
Run `python3 main.py`, images and labels.txt will generate at `output/default/`

Some optional arguments:
- num_img: how many images to generate
- output_dir: where to save the images
- corpus_dir: put txt file in corpus_dir
- corpus_mode: different corpus type have different load/get_sample method, see corresponding function for detail
- chars_file: chars not contained in chars_file will be filtered
- bg_dir: 50% image background are loaded from background image dir
- line: add underline, crop from table line, middle highlight
- noise: add gauss noise, uniform, salt/pepper noise, poisson noise

There are a lot of configs used in renderer.py, you should change it to meet your own requirements.

# Generate image using GPU
If you want to use GPU to speed up image generating, first compile opencv with CUDA.
[Compiling OpenCV with CUDA support](https://www.pyimagesearch.com/2016/07/11/compiling-opencv-with-cuda-support/)

Then build Cython part, and add `--gpu` options when run main.py
```
cd libs/gpu
python3 setup.py build_ext --inplace
```


# Todo
- [ ] refactor code, make more configurable
- [ ] pre check each font supportted chars 
- [ ] draw word on background image use Seamless cloning
- [ ] make char space configurable, currently it's rely on Pillow's implement
- [ ] word balance
- [ ] generate color text image
