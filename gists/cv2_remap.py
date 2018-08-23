"""
https://vlight.me/2018/06/25/OpenCV-Recipes-Genometric-Transformations/
"""
import cv2
import numpy as np
import math

img = cv2.imread('../imgs/debug_demo.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('../imgs/example1.jpg', cv2.IMREAD_GRAYSCALE)
img_output = np.zeros(img.shape, dtype=img.dtype)

rows, cols = img.shape

print(img.shape)

img_x = np.zeros((rows, cols), np.float32)
img_y = np.zeros((rows, cols), np.float32)

# 坐标映射
for y in range(rows):
    for x in range(cols):
        # 某一个位置的 y 值应该为那个位置的 y 值
        img_y[y, x] = y + int(16.0 * math.sin(2 * 3.14 * x / 200))
        # 某一个位置的 x 值应该为那个位置的 x 值
        img_x[y, x] = x

dst = cv2.remap(img, img_x, img_y, cv2.INTER_LINEAR)
cv2.imshow('remap', dst)
cv2.waitKey()

for i in range(rows):
    for j in range(cols):
        offset_x = 0
        # 300 决定了弧线的宽度，16决定了弧线的高度
        offset_y = int(6.0 * math.sin(2 * 3.14 * j / 300))
        if i + offset_y < rows:
            img_output[i, j] = img[(i + offset_y), j]
        else:
            img_output[i, j] = 0

cv2.imshow('Horizontal wave', img_output)
cv2.waitKey()

img_output = np.zeros(img.shape, dtype=img.dtype)

for i in range(rows):
    for j in range(cols):
        offset_x = int(28.0 * math.sin(2 * 3.14 * i / (2 * cols)))
        offset_y = 0
        if j + offset_x < cols:
            img_output[i, j] = img[i, (j + offset_x) % cols]
        else:
            img_output[i, j] = 0

cv2.imshow('Concave', img_output)
cv2.waitKey()
