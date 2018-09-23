import os
import random
import cv2
import numpy as np

from PIL import Image, ImageFont, ImageDraw

pointsize = 30
bg_name = "../data/bg/背景.png"
text = "Hello World!你好世界"
font = '../data/fonts/chn/msyh.ttc'
text_color = (0, 0, 0)

# Random bg
bg_high = random.uniform(220, 255)
bg_low = bg_high - random.uniform(1, 60)
bg = np.random.randint(bg_low, bg_high, (64, 512, 4)).astype(np.uint8)

# Draw text on
word_mask = Image.new('RGBA', (512, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(word_mask)
font = ImageFont.truetype(font, pointsize)
draw.text((0, 0), text, font=font, fill=text_color)
# img_gray = img.convert('L')
# img_gray.save('text.jpg')

pil_bg = Image.open(bg_name)
pil_bg = pil_bg.convert('RGBA')

# pil_bg = Image.fromarray(np.uint8(bg))
pil_bg.paste(word_mask, (0, 0), mask=word_mask)

pil_bg = pil_bg.convert('L')
pil_bg.save('test.jpg')
