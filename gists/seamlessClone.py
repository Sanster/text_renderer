from PIL import ImageFont, Image, ImageDraw
import numpy as np
import cv2

# bg_path = '/home/cwq/code/ocr_end2end/text_gen/bg/000000088218.jpg'
from gists.outline import draw_border_text

bg_path = '/home/cwq/code/text_renderer/data/bg/paper1.png'
font_path = '/home/cwq/code/ocr_end2end/text_gen/fonts/chn/msyh.ttc'

COLOR = 200
SEAMLESS_OFFSET = 9
BORDER_COLOR = COLOR + 20
BORDER_THICKNESS = 0.5
FONT_SIZE = 45

word = '测试图像的无缝融合'
font = ImageFont.truetype(font_path, FONT_SIZE)
offset = font.getoffset(word)
size = font.getsize(word)

print(offset)
word_height = size[1] - offset[1]
word_width = size[0]

# bg = np.ones((size[1] - offset[1], size[0] - offset[0]), np.uint8) * 255
bg = cv2.imread(bg_path)
bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
bg_height = bg.shape[0]
bg_width = bg.shape[1]

white_bg = np.ones((word_height + SEAMLESS_OFFSET, word_width + SEAMLESS_OFFSET)) * 255
text_img = Image.fromarray(np.uint8(white_bg))
draw = ImageDraw.Draw(text_img)

draw_border_text(draw=draw, text=word,
                 x=0 + SEAMLESS_OFFSET // 2,
                 y=0 - offset[1] + SEAMLESS_OFFSET // 2,
                 font=font, thickness=BORDER_THICKNESS, border_color=BORDER_COLOR,
                 text_color=COLOR)
# draw.text((0 + SEAMLESS_OFFSET // 2, 0 - offset[1] + SEAMLESS_OFFSET // 2), word, fill=COLOR, font=font)

bg_gray = Image.fromarray(np.uint8(bg_gray))
graw_draw = ImageDraw.Draw(bg_gray)

# graw_draw.text((text_x, text_y), word, fill=COLOR, font=font)
draw_border_text(draw=graw_draw, text=word,
                 x=(bg_width - word_width) // 2,
                 y=(bg_height - word_height) // 2,
                 font=font, thickness=BORDER_THICKNESS, border_color=BORDER_COLOR,
                 text_color=COLOR)

bg_gray.save('direct.jpg')

text_img.save('text.jpg')
text_img = np.array(text_img).astype(np.uint8)

text_mask = 255 * np.ones(text_img.shape, text_img.dtype)

# This is where the CENTER of the airplane will be placed
center = (bg_width // 2, bg_height // 2)

text_img_bgr = np.ones((text_img.shape[0], text_img.shape[1], 3), np.uint8)
cv2.cvtColor(text_img, cv2.COLOR_GRAY2BGR, text_img_bgr)

print(text_img_bgr.shape)
print(bg.shape)
print(text_mask.shape)
mixed_clone = cv2.seamlessClone(text_img_bgr, bg, text_mask, center, cv2.MONOCHROME_TRANSFER)

result = cv2.cvtColor(mixed_clone, cv2.COLOR_BGR2GRAY)
cv2.imwrite('seamless.jpg', result)
