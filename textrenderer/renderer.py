import math
import random
import numpy as np
import cv2
from PIL import ImageFont, Image, ImageDraw
from tenacity import retry

import libs.math_utils as math_utils
from libs.utils import draw_box, draw_bbox, prob, apply
from libs.timer import Timer
from textrenderer.liner import Liner
from textrenderer.noiser import Noiser
import libs.font_utils as font_utils

# noinspection PyMethodMayBeStatic
from textrenderer.remaper import Remaper


class Renderer(object):
    def __init__(self, corpus, fonts, bgs, cfg, width=256, height=32,
                 clip_max_chars=False, debug=False, gpu=False, strict=False):
        self.corpus = corpus
        self.fonts = fonts
        self.bgs = bgs
        self.out_width = width
        self.out_height = height
        self.clip_max_chars = clip_max_chars
        self.max_chars = math.floor(width / 4) - 1
        self.debug = debug
        self.gpu = gpu
        self.strict = strict
        self.cfg = cfg

        self.timer = Timer()
        self.liner = Liner(cfg)
        self.noiser = Noiser(cfg)
        self.remaper = Remaper(cfg)

        self.create_kernals()

        if self.strict:
            self.font_unsupport_chars = font_utils.get_unsupported_chars(self.fonts, corpus.chars_file)

    def gen_img(self, img_index):
        word, font, word_size = self.pick_font(img_index)
        self.dmsg("after pick font")

        # Background's height should much larger than raw word image's height,
        # to make sure we can crop full word image after apply perspective
        bg = self.gen_bg(width=word_size[0] * 8, height=word_size[1] * 8)
        word_img, text_box_pnts, word_color = self.draw_text_on_bg(word, font, bg)
        self.dmsg("After draw_text_on_bg")

        if apply(self.cfg.crop):
            text_box_pnts = self.apply_crop(text_box_pnts, self.cfg.crop)

        if apply(self.cfg.line):
            word_img, text_box_pnts = self.liner.apply(word_img, text_box_pnts, word_color)
            self.dmsg("After draw line")

        if self.debug:
            word_img = draw_box(word_img, text_box_pnts, (0, 255, 155))

        if apply(self.cfg.curve):
            word_img, text_box_pnts = self.remaper.apply(word_img, text_box_pnts, word_color)
            self.dmsg("After remapping")

        if self.debug:
            word_img = draw_box(word_img, text_box_pnts, (155, 255, 0))

        word_img, img_pnts_transformed, text_box_pnts_transformed = \
            self.apply_perspective_transform(word_img, text_box_pnts,
                                             max_x=self.cfg.perspective_transform.max_x,
                                             max_y=self.cfg.perspective_transform.max_y,
                                             max_z=self.cfg.perspective_transform.max_z,
                                             gpu=self.gpu)

        self.dmsg("After perspective transform")

        if self.debug:
            _, crop_bbox = self.crop_img(word_img, text_box_pnts_transformed)
            word_img = draw_bbox(word_img, crop_bbox, (255, 0, 0))
        else:
            word_img, crop_bbox = self.crop_img(word_img, text_box_pnts_transformed)

        self.dmsg("After crop_img")

        if apply(self.cfg.noise):
            word_img = np.clip(word_img, 0., 255.)
            word_img = self.noiser.apply(word_img)
            self.dmsg("After noiser")

        blured = False
        if apply(self.cfg.blur):
            blured = True
            word_img = self.apply_blur_on_output(word_img)
            self.dmsg("After blur")

        if not blured:
            if apply(self.cfg.prydown):
                word_img = self.apply_prydown(word_img)
                self.dmsg("After prydown")

        word_img = np.clip(word_img, 0., 255.)

        if apply(self.cfg.reverse_color):
            word_img = self.reverse_img(word_img)
            self.dmsg("After reverse_img")

        if apply(self.cfg.emboss):
            word_img = self.apply_emboss(word_img)
            self.dmsg("After emboss")

        if apply(self.cfg.sharp):
            word_img = self.apply_sharp(word_img)
            self.dmsg("After sharp")

        return word_img, word

    def dmsg(self, msg):
        if self.debug:
            print(msg)

    def random_xy_offset(self, src_height, src_width, dst_height, dst_width):
        """
        Get random left-top point for putting a small rect in a large rect.
        Normally dst_height>src_height and dst_width>src_width
        """
        y_max_offset = 0
        if dst_height > src_height:
            y_max_offset = dst_height - src_height

        x_max_offset = 0
        if dst_width > src_width:
            x_max_offset = dst_width - src_width

        y_offset = 0
        if y_max_offset != 0:
            y_offset = random.randint(0, y_max_offset)

        x_offset = 0
        if x_max_offset != 0:
            x_offset = random.randint(0, x_max_offset)

        return x_offset, y_offset

    def crop_img(self, img, text_box_pnts_transformed):
        """
        Crop text from large input image
        :param img: image to crop
        :param text_box_pnts_transformed: text_bbox_pnts after apply_perspective_transform
        :return:
            dst: image with desired output size, height=32, width=flags.img_width
            crop_bbox: bounding box on input image
        """
        bbox = cv2.boundingRect(text_box_pnts_transformed)
        bbox_width = bbox[2]
        bbox_height = bbox[3]

        # Output shape is (self.out_width, self.out_height)
        # We randomly put bounding box of transformed text in the output shape
        # so the max value of dst_height is out_height

        # TODO: If rotate angle(z) of text is too big, text will become very small,
        # we should do something to prevent text too small

        # dst_height and dst_width is used to leave some padding around text bbox
        dst_height = random.randint(self.out_height // 4 * 3, self.out_height)

        if self.out_width == 0:
            scale = bbox_height / dst_height
        else:
            dst_width = self.out_width
            scale = max(bbox_height / dst_height, bbox_width / self.out_width)

        s_bbox_width = math.ceil(bbox_width / scale)
        s_bbox_height = math.ceil(bbox_height / scale)

        if self.out_width == 0:
            padding = random.randint(s_bbox_width // 10, s_bbox_width // 8)
            dst_width = s_bbox_width + padding * 2

        s_bbox = (np.around(bbox[0] / scale),
                  np.around(bbox[1] / scale),
                  np.around(bbox[2] / scale),
                  np.around(bbox[3] / scale))

        x_offset, y_offset = self.random_xy_offset(s_bbox_height, s_bbox_width, self.out_height, dst_width)

        dst_bbox = (
            self.int_around((s_bbox[0] - x_offset) * scale),
            self.int_around((s_bbox[1] - y_offset) * scale),
            self.int_around(dst_width * scale),
            self.int_around(self.out_height * scale)
        )

        # It's important do crop first and than do resize for speed consider
        dst = img[dst_bbox[1]:dst_bbox[1] + dst_bbox[3], dst_bbox[0]:dst_bbox[0] + dst_bbox[2]]

        dst = cv2.resize(dst, (dst_width, self.out_height), interpolation=cv2.INTER_CUBIC)

        return dst, dst_bbox

    def int_around(self, val):
        return int(np.around(val))

    def get_word_color(self, bg, text_x, text_y, word_height, word_width):
        """
        Only use word roi area to get word color
        """
        offset = 10
        ymin = text_y - offset
        ymax = text_y + word_height + offset
        xmin = text_x - offset
        xmax = text_x + word_width + offset

        word_roi_bg = bg[ymin: ymax, xmin: xmax]

        bg_mean = int(np.mean(word_roi_bg) * (2 / 3))
        word_color = random.randint(0, bg_mean)
        return word_color

    def draw_text_on_bg(self, word, font, bg):
        """
        Draw word in the center of background
        :param word: word to draw
        :param font: font to draw word
        :param bg: background numpy image
        :return:
            np_img: word image
            text_box_pnts: left-top, right-top, right-bottom, left-bottom
        """
        bg_height = bg.shape[0]
        bg_width = bg.shape[1]

        word_size = self.get_word_size(font, word)
        word_height = word_size[1]
        word_width = word_size[0]

        offset = font.getoffset(word)

        pil_img = Image.fromarray(np.uint8(bg))
        draw = ImageDraw.Draw(pil_img)

        # Draw text in the center of bg
        text_x = int((bg_width - word_width) / 2)
        text_y = int((bg_height - word_height) / 2)

        word_color = self.get_word_color(bg, text_x, text_y, word_height, word_width)

        if apply(self.cfg.random_space):
            text_x, text_y, word_width, word_height = self.draw_text_with_random_space(draw, font, word, word_color,
                                                                                       bg_width, bg_height)
            np_img = np.array(pil_img).astype(np.float32)
        else:
            if apply(self.cfg.seamless_clone):
                np_img = self.draw_text_seamless(font, bg, word, word_color, word_height, word_width, offset)
            else:
                self.draw_text_wrapper(draw, word, text_x - offset[0], text_y - offset[1], font, word_color)
                # draw.text((text_x - offset[0], text_y - offset[1]), word, fill=word_color, font=font)

                np_img = np.array(pil_img).astype(np.float32)

        text_box_pnts = [
            [text_x, text_y],
            [text_x + word_width, text_y],
            [text_x + word_width, text_y + word_height],
            [text_x, text_y + word_height]
        ]

        return np_img, text_box_pnts, word_color

    def draw_text_seamless(self, font, bg, word, word_color, word_height, word_width, offset):
        # For better seamlessClone
        seamless_offset = 6

        # Draw text on a white image, than draw it on background
        white_bg = np.ones((word_height + seamless_offset, word_width + seamless_offset)) * 255
        text_img = Image.fromarray(np.uint8(white_bg))
        draw = ImageDraw.Draw(text_img)

        # draw.text((0 + seamless_offset // 2, 0 - offset[1] + seamless_offset // 2), word,
        #           fill=word_color, font=font)

        self.draw_text_wrapper(draw, word,
                               0 + seamless_offset // 2,
                               0 - offset[1] + seamless_offset // 2,
                               font, word_color)

        # assume whole text_img as mask
        text_img = np.array(text_img).astype(np.uint8)
        text_mask = 255 * np.ones(text_img.shape, text_img.dtype)

        # This is where the CENTER of the airplane will be placed
        center = (bg.shape[1] // 2, bg.shape[0] // 2)

        # opencv seamlessClone require bgr image
        text_img_bgr = np.ones((text_img.shape[0], text_img.shape[1], 3), np.uint8)
        bg_bgr = np.ones((bg.shape[0], bg.shape[1], 3), np.uint8)
        cv2.cvtColor(text_img, cv2.COLOR_GRAY2BGR, text_img_bgr)
        cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR, bg_bgr)

        flag = np.random.choice([
            cv2.NORMAL_CLONE,
            cv2.MIXED_CLONE,
            cv2.MONOCHROME_TRANSFER
        ])

        mixed_clone = cv2.seamlessClone(text_img_bgr, bg_bgr, text_mask, center, flag)

        np_img = cv2.cvtColor(mixed_clone, cv2.COLOR_BGR2GRAY)

        return np_img

    def draw_text_with_random_space(self, draw, font, word, word_color, bg_width, bg_height):
        """ If random_space applied, text_x, text_y, word_width, word_height may change"""
        width = 0
        height = 0
        chars_size = []
        y_offset = 10 ** 5
        for c in word:
            size = font.getsize(c)
            chars_size.append(size)

            width += size[0]
            # set max char height as word height
            if size[1] > height:
                height = size[1]

            # Min chars y offset as word y offset
            # Assume only y offset
            c_offset = font.getoffset(c)
            if c_offset[1] < y_offset:
                y_offset = c_offset[1]

        char_space_width = int(height * np.random.uniform(self.cfg.random_space.min, self.cfg.random_space.max))

        width += (char_space_width * (len(word) - 1))

        text_x = int((bg_width - width) / 2)
        text_y = int((bg_height - height) / 2)

        c_x = text_x
        c_y = text_y

        for i, c in enumerate(word):
            # self.draw_text_wrapper(draw, c, c_x, c_y - y_offset, font, word_color, force_text_border)
            draw.text((c_x, c_y - y_offset), c, fill=word_color, font=font)

            c_x += (chars_size[i][0] + char_space_width)

        return text_x, text_y, width, height

    def draw_text_wrapper(self, draw, text, x, y, font, text_color):
        """
        :param x/y: 应该是移除了 offset 的
        """
        if apply(self.cfg.text_border):
            self.draw_border_text(draw, text, x, y, font, text_color)
        else:
            draw.text((x, y), text, fill=text_color, font=font)

    def draw_border_text(self, draw, text, x, y, font, text_color):
        """
        :param x/y: 应该是移除了 offset 的
        """
        # thickness larger than 1 may give bad border result
        thickness = 1

        choices = []
        p = []
        if self.cfg.text_border.light.enable:
            choices.append(0)
            p.append(self.cfg.text_border.light.fraction)
        if self.cfg.text_border.dark.enable:
            choices.append(1)
            p.append(self.cfg.text_border.dark.fraction)

        light_or_dark = np.random.choice(choices, p=p)

        if light_or_dark == 0:
            border_color = text_color + np.random.randint(0, 255 - text_color - 1)
        elif light_or_dark == 1:
            border_color = text_color - np.random.randint(0, text_color + 1)

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

    def gen_bg(self, width, height):
        if apply(self.cfg.img_bg):
            bg = self.gen_bg_from_image(int(width), int(height))
        else:
            bg = self.gen_rand_bg(int(width), int(height))
        return bg

    def gen_rand_bg(self, width, height):
        """
        Generate random background
        """
        bg_high = random.uniform(220, 255)
        bg_low = bg_high - random.uniform(1, 60)

        bg = np.random.randint(bg_low, bg_high, (height, width)).astype(np.uint8)

        bg = self.apply_gauss_blur(bg)

        return bg

    def gen_bg_from_image(self, width, height):
        """
        Resize background, let bg_width>=width, bg_height >=height, and random crop from resized background
        """
        assert width > height

        bg = random.choice(self.bgs)

        scale = max(width / bg.shape[1], height / bg.shape[0])

        out = cv2.resize(bg, None, fx=scale, fy=scale)

        x_offset, y_offset = self.random_xy_offset(height, width, out.shape[0], out.shape[1])

        out = out[y_offset:y_offset + height, x_offset:x_offset + width]

        out = self.apply_gauss_blur(out, ks=[7, 11, 13, 15, 17])

        bg_mean = int(np.mean(out))

        # TODO: find a better way to deal with background
        # alpha = 255 / bg_mean  # 对比度
        # beta = np.random.randint(bg_mean // 4, bg_mean // 2)  # 亮度
        # out = np.uint8(np.clip((alpha * out + beta), 0, 255))

        return out

    @retry
    def pick_font(self, img_index):
        """
        :param img_index when use list corpus, this param is used
        :return:
            font: truetype
            size: word size, removed offset (width, height)
        """
        word = self.corpus.get_sample(img_index)

        if self.clip_max_chars and len(word) > self.max_chars:
            word = word[:self.max_chars]

        font_path = random.choice(self.fonts)

        if self.strict:
            unsupport_chars = self.font_unsupport_chars[font_path]
            for c in word:
                if c == ' ':
                    continue
                if c in unsupport_chars:
                    print('Retry pick_font(), \'%s\' contains chars \'%s\' not supported by font %s' % (
                        word, c, font_path))
                    raise Exception

        # Font size in point
        font_size = random.randint(self.cfg.font_size.min, self.cfg.font_size.max)
        font = ImageFont.truetype(font_path, font_size)

        return word, font, self.get_word_size(font, word)

    def get_word_size(self, font, word):
        """
        Get word size removed offset
        :param font: truetype
        :param word:
        :return:
            size: word size, removed offset (width, height)
        """
        offset = font.getoffset(word)
        size = font.getsize(word)
        size = (size[0] - offset[0], size[1] - offset[1])
        return size

    def apply_perspective_transform(self, img, text_box_pnts, max_x, max_y, max_z, gpu=False):
        """
        Apply perspective transform on image
        :param img: origin numpy image
        :param text_box_pnts: four corner points of text
        :param x: max rotate angle around X-axis
        :param y: max rotate angle around Y-axis
        :param z: max rotate angle around Z-axis
        :return:
            dst_img:
            dst_img_pnts: points of whole word image after apply perspective transform
            dst_text_pnts: points of text after apply perspective transform
        """

        x = math_utils.cliped_rand_norm(0, max_x)
        y = math_utils.cliped_rand_norm(0, max_y)
        z = math_utils.cliped_rand_norm(0, max_z)

        # print("x: %f, y: %f, z: %f" % (x, y, z))

        transformer = math_utils.PerspectiveTransform(x, y, z, scale=1.0, fovy=50)

        dst_img, M33, dst_img_pnts = transformer.transform_image(img, gpu)
        dst_text_pnts = transformer.transform_pnts(text_box_pnts, M33)

        return dst_img, dst_img_pnts, dst_text_pnts

    def apply_blur_on_output(self, img):
        if prob(0.5):
            return self.apply_gauss_blur(img, [3, 5])
        else:
            return self.apply_norm_blur(img)

    def apply_gauss_blur(self, img, ks=None):
        if ks is None:
            ks = [7, 9, 11, 13]
        ksize = random.choice(ks)

        sigmas = [0, 1, 2, 3, 4, 5, 6, 7]
        sigma = 0
        if ksize <= 3:
            sigma = random.choice(sigmas)
        img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        return img

    def apply_norm_blur(self, img, ks=None):
        # kernel == 1, the output image will be the same
        if ks is None:
            ks = [2, 3]
        kernel = random.choice(ks)
        img = cv2.blur(img, (kernel, kernel))
        return img

    def apply_prydown(self, img):
        """
        模糊图像，模拟小图片放大的效果
        """
        scale = random.uniform(1, self.cfg.prydown.max_scale)
        height = img.shape[0]
        width = img.shape[1]

        out = cv2.resize(img, (int(width / scale), int(height / scale)), interpolation=cv2.INTER_AREA)
        return cv2.resize(out, (width, height), interpolation=cv2.INTER_AREA)

    def reverse_img(self, word_img):
        offset = np.random.randint(-10, 10)
        return 255 + offset - word_img

    def create_kernals(self):
        self.emboss_kernal = np.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ])

        self.sharp_kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ])

    def apply_emboss(self, word_img):
        return cv2.filter2D(word_img, -1, self.emboss_kernal)

    def apply_sharp(self, word_img):
        return cv2.filter2D(word_img, -1, self.sharp_kernel)

    def apply_crop(self, text_box_pnts, crop_cfg):
        """
        Random crop text box height top or bottom, we don't need image information in this step, only change box pnts
        :param text_box_pnts: bbox of text [left-top, right-top, right-bottom, left-bottom]
        :param crop_cfg:
        :return:
            croped_text_box_pnts
        """
        height = abs(text_box_pnts[0][1] - text_box_pnts[3][1])
        scale = float(height) / float(self.out_height)

        croped_text_box_pnts = text_box_pnts

        if prob(0.5):
            top_crop = int(random.randint(crop_cfg.top.min, crop_cfg.top.max) * scale)
            self.dmsg("top crop %d" % top_crop)
            croped_text_box_pnts[0][1] += top_crop
            croped_text_box_pnts[1][1] += top_crop
        else:
            bottom_crop = int(random.randint(crop_cfg.bottom.min, crop_cfg.bottom.max) * scale)
            self.dmsg("bottom crop %d " % bottom_crop)
            croped_text_box_pnts[2][1] -= bottom_crop
            croped_text_box_pnts[3][1] -= bottom_crop

        return croped_text_box_pnts
