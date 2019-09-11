#!/usr/bin/env python3
# -*- coding: utf-8 -*-

fonts = r"E:\deeplearn\OCR\Font-chi-筛检"
smuDir =r'E:\deeplearn\OCR2\ocr-data\zhiwen'
bgDir = r'E:\deeplearn\OCR2\ocr-data\bg'

#smuDir ='NISTSpecialDatabase4GrayScaleImagesofFIGS'
#bgDir = 'bg'
#fonts = "fonts"


underline_percent = 30
gauss_percent = 30  
gauss_range=[0,0.6]
distort_percent = 30
geometry_percent = 30
smudginess_percent = 30
rand_pading_percent = 30
add_biaodian_percent = 0
skewing_percent = 10
skewing_range = (-2, 3)
font_gray=[0,0]


distortParamList = [
    (3.0, 10.0),
    (2.0, 10.0),
    (1.0, 10.0)
]

geometryParamlist = [
    (0.0001, 0.00003),
    (0.002, 0.0003),
    (0.0002, 0.00005),
    (0.0003, 0.00002),
    (0.0005, 0.00006)
]
