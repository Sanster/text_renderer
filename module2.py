


# 增加指纹的
import os
import random

import cv2
import math
import numpy
import numpy as np
from PIL import ImageFont, Image, ImageDraw
from numpy import asarray, amax
from scipy.ndimage import filters, measurements, interpolation
from numpy.random import randn
import matplotlib.pyplot as plt 



# 颜色减淡操作
def dodge(gray, factor=1.0):
    return np.minimum(gray+100*factor, 255)
def reverse(gray):
    return 255 - gray

def rotat(img):
    img = reverse(img)
    rows, cols = numpy.shape(img)
    angle = random.randint(-30, 30)
    m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img = cv2.warpAffine(img, m, (cols, rows))
    img = reverse(img)
    return img


def get_smudginess(filename, size=(60, 120)):

    smu = cv2.imread(filename, 0)
    smu = cv2.resize(smu, size,interpolation = cv2.INTER_NEAREST)
    smu = asarray(smu, 'f')
    smu = dodge(smu, random.uniform(0.4, 0.9))
    smu = rotat(smu)
    return smu


import glob
dir = r'C:\jianweidata\ocr\text_renderer\textrenderer\tmp\zhiwen'
files = glob.glob(os.path.join(dir,'*.png'))
for idx,file in enumerate(files):
    smu = get_smudginess(file)
    cv2.imwrite(os.path.join(dir,str(idx)+'.jpg'),smu)