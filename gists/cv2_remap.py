"""
https://vlight.me/2018/06/25/OpenCV-Recipes-Genometric-Transformations/
"""
import cv2
import numpy as np
import math

from PIL import ImageDraw, Image, ImageFont


def draw_four_vectors(img, line, color=(0, 255, 0)):
    """
    :param line: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        矩形四点坐标的顺序： left-top, right-top, right-bottom, left-bottom
    """
    img = cv2.line(img, (line[0][0], line[0][1]), (line[1][0], line[1][1]), color)
    img = cv2.line(img, (line[1][0], line[1][1]), (line[2][0], line[2][1]), color)
    img = cv2.line(img, (line[2][0], line[2][1]), (line[3][0], line[3][1]), color)
    img = cv2.line(img, (line[3][0], line[3][1]), (line[0][0], line[0][1]), color)
    return img


h = 1000
w = 1000
text = "关于那次灾难的奇特"
font_path = './data/fonts/chn/msyh.ttc'
font_size = 30

font = ImageFont.truetype(font_path, font_size)

# get text bounding box
offset = font.getoffset(text)
size = font.getsize(text)
size = (size[0] - offset[0], size[1] - offset[1])

text_x = w // 2
text_y = h // 2
word_width = size[0]
word_height = size[1]

text_box_pnts = [
    [text_x, text_y],
    [text_x + word_width, text_y],
    [text_x + word_width, text_y + word_height],
    [text_x, text_y + word_height]
]

img = np.ones((h, w, 3), np.uint8) * 255
img = draw_four_vectors(img, text_box_pnts)

pil_img = Image.fromarray(np.uint8(img))
draw = ImageDraw.Draw(pil_img)

draw.text((text_x - offset[0], text_y - offset[1]), text, fill=0, font=font)

img = np.array(pil_img).astype(np.float32)
img_x = np.zeros((h, w), np.float32)
img_y = np.zeros((h, w), np.float32)

period = 360  # degree
max_val = 18


def remap_y(x):
    return int(max_val * math.sin(2 * 3.14 * x / period))


# 坐标映射
for y in range(h):
    for x in range(w):
        # 某一个位置的 y 值应该为那个位置的 y 值
        img_y[y, x] = y + remap_y(x)
        # 某一个位置的 x 值应该为那个位置的 x 值
        img_x[y, x] = x

# remap text box pnts
dst = cv2.remap(img, img_x, img_y, cv2.INTER_CUBIC)

# 对 text_x 和 text_x + word_width 之间的所有值进行 remap
# 找到 remap 以后的 ymin 和 ymax
remap_y_offset_min_x = min(list(range(text_x, text_x + word_width)), key=lambda x: remap_y(x))
remap_y_offset_max_x = max(list(range(text_x, text_x + word_width)), key=lambda x: remap_y(x))

remap_y_offset_min = remap_y(remap_y_offset_min_x)
remap_y_offset_max = remap_y(remap_y_offset_max_x)

print("remap_y_offser_min_x={}, remap_y_offset_min={}".format(remap_y_offset_min_x, remap_y_offset_min))
print("remap_y_offser_max_x={}, remap_y_offset_max={}".format(remap_y_offset_max_x, remap_y_offset_max))

remaped_text_box_pnts = [
    [text_x, text_y + remap_y_offset_min],
    [text_x + word_width, text_y + remap_y_offset_min],
    [text_x + word_width, text_y + word_height + remap_y_offset_max],
    [text_x, text_y + word_height + remap_y_offset_max]
]

img = draw_four_vectors(dst, remaped_text_box_pnts, color=(0, 0, 255))

cv2.imshow('cv.remap() sin', dst)
cv2.waitKey()

# for i in range(rows):
#     for j in range(cols):
#         offset_x = 0
#         # 300 决定了弧线的宽度，16决定了弧线的高度
#         offset_y = int(6.0 * math.sin(2 * 3.14 * j / 300))
#         if i + offset_y < rows:
#             img_output[i, j] = img[(i + offset_y), j]
#         else:
#             img_output[i, j] = 0
#
# cv2.imshow('Horizontal wave', img_output)
# cv2.waitKey()

# img_output = np.zeros(img.shape, dtype=img.dtype)
#
# for i in range(rows):
#     for j in range(cols):
#         offset_x = int(28.0 * math.sin(2 * 3.14 * i / (2 * cols)))
#         offset_y = 0
#         if j + offset_x < cols:
#             img_output[i, j] = img[i, (j + offset_x) % cols]
#         else:
#             img_output[i, j] = 0
#
# cv2.imshow('Concave', img_output)
# cv2.waitKey()
