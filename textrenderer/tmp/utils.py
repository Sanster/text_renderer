#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import config


def str_shuffle(s):
    return ''.join(random.sample(s, len(s)))


def get_dic_lines(alphabet):
    lines = []
    s = str_shuffle(alphabet)
    start = 0
    end = 0
    l = random.randint(3, 6)
    end += l
    length = len(s)
    while end < length:
        print(s[start:end])
        lines.append(s[start:end])
        start = end
        end += random.randint(3, 6)
    if end != length:
        lines.append(s[start:] + "".join(random.sample(alphabet, end-length)))
    return lines


def load_fonts(fonts_dir):

    font_files = []
    for root, dirs, files in os.walk(fonts_dir):
        for name in files:
            font_files.append(os.path.join(root, name))
    fonts = []
    for fontFile in font_files:
        fonts.append(ImageFont.truetype(fontFile, 48))
    return fonts


def load_fonts_path(fonts_dir):
    font_files = []
    for root, dirs, files in os.walk(fonts_dir):
        for name in files:
            font_files.append(os.path.join(root, name))

    return font_files


def load_font(file):
    return ImageFont.truetype(file, 48)


def load_fonts_size(file, size):
    fonts = []
    start, end = size
    for i in range(start, end):
        fonts.append(ImageFont.truetype(file, i))
    return fonts


def load_bgs(bg_dir):
    bg_files = []
    for root, dirs, files in os.walk(bg_dir):
        for name in files:
            bg_files.append(os.path.join(root, name))
    return bg_files


def load_smudginess(smu_dir):
    f = []
    for root, dirs, files in os.walk(smu_dir):
        for name in files:
            if os.path.splitext(name)[1] == '.png':
                f.append(os.path.join(root,name))
    return f


def load_bg(path):
    bg = Image.open(path);
    bg = bg.convert("RGB")
    return bg.resize((1200, 300), Image.ANTIALIAS)


def load_textfiles(dir_path):
    print("load text file start")
    words = []
    for root, dirs, files in os.walk(dir_path):
        for name in files:
            w = load_words(os.path.join(root, name))
            if w:
                words.extend(w)
    return words


def load_words(file):
    print("load text : ", file)
    try:
        with open(file) as fr:
            return [line.strip() for line in fr]
    except Exception as e:
        print(e)


def rgeometry(image, eps=0.0001, delta=0.00003):
    m = numpy.array([[1 + eps * randn(), 0.0], [eps * randn(), 1.0 + eps * randn()]])
    w, h = image.shape
    c = numpy.array([w / 2.0, h / 2])
    d = c - numpy.dot(m, c) + numpy.array([randn() * delta, numpy.random.randn() * delta])
    return interpolation.affine_transform(image, m, offset=d, order=1, mode='constant', cval=image[0, 0])


def rdistort(image, distort=3.0, dsigma=10.0, cval=0):

    h, w = image.shape
    hs = randn(h, w)
    ws = randn(h, w)
    hs = filters.gaussian_filter(hs, dsigma)
    ws = filters.gaussian_filter(ws, dsigma)
    hs *= distort/amax(hs)
    ws *= distort/amax(ws)

    def f(p):
        return p[0]+hs[p[0], p[1]], p[1]+ws[p[0], p[1]]

    return interpolation.geometric_transform(image, f, output_shape=(h, w), order=1, mode='constant', cval=cval)


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


def get_smudginess(filename, size=(80, 120)):

    smu = cv2.imread(filename, 0)
    smu = cv2.resize(smu, size)
    smu = asarray(smu, 'f')
    smu = dodge(smu, random.uniform(0.8, 1.1))
    smu = rotat(smu)
    return smu


def add_smudginess(img, smu, pos):
    return merge_pic(img, smu, pos)


def merge_pic(target, source, pos):

    w, h = source.shape
    x, y = pos
    cv2.bitwise_not(source, source)
    image_roi = target[x:x + w, y:y + h]
    print(image_roi.shape)

    cv2.bitwise_not(image_roi, image_roi)
    rest = cv2.bitwise_or(source, image_roi)

    # 结果取反
    cv2.bitwise_not(rest, rest)
    cv2.bitwise_not(source, source)
    target[x:x + w, y:y + h] = rest
    return target

#size=top,bottom,left,right
def crop_with_size(image, size, pad=(1, 1, 1, 1)):
    y1, y2, x1, x2 = pad
    r0, r1, c0, c1 = size
    image = image[r0 - y1:r1 + y2, c0 - x1:c1 + x2]
    return image


def sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)


#def add_gauss(img, level):
#    return cv2.GaussianBlur(img, (level * 2 + 1, level * 2 + 1),1.5);
def add_gauss(img, gauss):
    return cv2.GaussianBlur(img, (0, 0),gauss);



def gen_line(bg, fonts, font_gray,text,  underline=False, gauss=0, distort_param=None, geometry_param=None, smudginess=None, skewing=None, outpadding=(0, 0, 0, 0,)):

    text_img = Image.new('RGBA', (3000, 300))
    draw = ImageDraw.Draw(text_img)

    (x, y, w, h) = draw_line(draw, fonts,font_gray, text, underline)

    crop_size = (y, y + h, x, x + w)
    y0, y1, x0, x1 = crop_size
    text_img = text_img.crop((x0, y0, x1, y1+random.randint(0,2)))

    w, h = text_img.size
    if skewing:
        start, end = skewing
        skewing_angle = random.randint(start, end)
        if skewing_angle < 0:
            skewing_angle += 360
        text_img = text_img.rotate(skewing_angle, expand=1)
        w, h = text_img.size

    text_img = text_img.resize((getWidth(h, w), 32))
    w, h = text_img.size
    max_width = 216

    if w > max_width:
        return None
    left = random.randint(30, bg.size[0] - max_width - 80 - 1)
    top = random.randint(10, bg.size[1] - h - 5)

    #32*max_width

    left_offset = random.randint(0, max_width - w)
    top_offset = random.randint(0, 4)

    random_left = left + left_offset
    random_top = top + top_offset
    bg.paste(text_img, (random_left, random_top), mask=text_img)
    image = bg.convert("L")
    im = asarray(image, 'f')

    if distort_param:
        distort, dsigma = distort_param
        cval = amax(im)
        im = rdistort(im, distort=distort, dsigma=dsigma, cval=cval)

    crop_size = (top, top + 32, left, left + max_width)
    if smudginess:
        smu = get_smudginess(smudginess)

        y0, _, x0, _ = crop_size
        x, y = (random.randint(x0-25, x0 + w), random.randint(0, 60))
        im = add_smudginess(im, smu, (y, x))
    im = crop_with_size(im, crop_size, outpadding)

    if geometry_param:
        eps, delta = geometry_param
        im = rgeometry(im, eps=eps, delta=delta)

    if gauss > 0:
        im = add_gauss(im, gauss)
    elif gauss < 0:
        im = sharpen(im)
    else:
        pass
    return im


def draw_underlined_text(draw, pos, text, font, font_gray,linesize=3, gap=2,  **options):

    bbox = draw_line_text(draw, pos, text, font,font_gray, **options)
    _, _, width, height = bbox
    lx, ly = pos[0], pos[1] + height

    draw.line((lx, ly + gap, lx + width, ly + gap), width=linesize, fill=(font_gray,font_gray,font_gray),  **options)
    return bbox[0], bbox[1], bbox[2], bbox[3] + gap + linesize + 1


def draw_line_text(draw, pos, text, fonts,font_gray, **options):
    x, y = pos
    h_max = 0
    for c in text:
        font = random.choice(fonts)
        w, h = font.getsize(c)
        if h < 42:
            h_padding = random.randint(0, int((48-h)/2))
        else:
            h_padding=0
        h_max = max(h_max, h + h_padding)

        draw.text((x, h_padding + y), c, font=font, fill=(font_gray,font_gray,font_gray), **options)
        x += w
    lx, ly = pos[0], pos[1]
    return lx, ly, x - lx, h_max+2


def draw_line(draw, fonts,font_gray, text, underline, **options):
    start = (40, 40)
    if underline:
        gap = random.randint(-2, 3)
        line_size = random.randint(1, 3)
        bbox = draw_underlined_text(draw, start, text, fonts,font_gray, linesize=line_size, gap=gap,
                                    **options)
        return bbox
    else:
        return draw_line_text(draw, start, text, fonts,font_gray, **options)


def random_str(charset, size):
    return ''.join(random.sample(charset, size))


def getWidth(h, w):
    high = 28
    rate = high / h
    width = int(np.ceil(rate * w))
    return width + 4 - width % 4