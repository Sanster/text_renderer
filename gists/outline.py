"""
https://mail.python.org/pipermail/image-sig/2009-May/005681.html
"""

import os
from PIL import Image, ImageFont, ImageDraw

x, y = 10, 10

fname1 = "../data/bg/背景.png"
im = Image.open(fname1)
pointsize = 25
thickness = 2
fillcolor = "red"
shadowcolor = "yellow"

text = "Hello World!你好世界"

font = '../data/fonts/chn/msyh.ttc'
draw = ImageDraw.Draw(im)
font = ImageFont.truetype(font, pointsize)


def draw_border_text(draw, text, x, y, font, thickness, border_color, text_color):
    # thin border
    draw.text((x - thickness, y), text, font=font, fill=border_color)
    draw.text((x + thickness, y), text, font=font, fill=border_color)
    draw.text((x, y - thickness), text, font=font, fill=border_color)
    draw.text((x, y + thickness), text, font=font, fill=border_color)

    # thicker border
    draw.text((x - thickness, y - thickness), text, font=font, fill=border_color)
    draw.text((x + thickness, y - thickness), text, font=font, fill=border_color)
    draw.text((x - thickness, y + thickness), text, font=font, fill=border_color)
    draw.text((x + thickness, y + thickness), text, font=font, fill=border_color)

    # now draw the text over it
    draw.text((x, y), text, font=font, fill=text_color)


fname2 = "test2.png"
im.save(fname2)

im.convert('LA').save('gray.png')
