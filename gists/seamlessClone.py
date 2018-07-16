from PIL import ImageFont, Image, ImageDraw
import numpy as np
import cv2

# bg_path = '/home/cwq/code/ocr_end2end/text_gen/bg/000000088218.jpg'
bg_path = '/home/cwq/code/text_renderer/data/bg/paper1.png'
font_path = '/home/cwq/code/ocr_end2end/text_gen/fonts/chn/msyh.ttc'

COLOR = 200
SEAMLESS_OFFSET = 6

word = '测试图像的无缝融合'
font = ImageFont.truetype(font_path, 25)
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
draw.text((0 + SEAMLESS_OFFSET // 2, 0 - offset[1] + SEAMLESS_OFFSET // 2), word, fill=COLOR, font=font)

bg_gray = Image.fromarray(np.uint8(bg_gray))
graw_draw = ImageDraw.Draw(bg_gray)
graw_draw.text((bg_width // 2 - word_width // 2, bg_height // 2 - word_height // 2), word, fill=COLOR, font=font)
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
cv2.imwrite('test.jpg', result)
