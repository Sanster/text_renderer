import random
import cv2
import numpy as np


class Remaper(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, word_img, text_box_pnts, word_color):
        """
        :param word_img:  word image with big background
        :param text_box_pnts: left-top, right-top, right-bottom, left-bottom of text word
        :return:
        """
        max_val = np.random.uniform(self.cfg.curve.min, self.cfg.curve.max)

        h = word_img.shape[0]
        w = word_img.shape[1]

        img_x = np.zeros((h, w), np.float32)
        img_y = np.zeros((h, w), np.float32)

        xmin = text_box_pnts[0][0]
        xmax = text_box_pnts[1][0]
        ymin = text_box_pnts[0][1]
        ymax = text_box_pnts[2][1]

        remap_y_min = ymin
        remap_y_max = ymax

        for y in range(h):
            for x in range(w):
                remaped_y = y + self._remap_y(x, max_val)

                if y == ymin:
                    if remaped_y < remap_y_min:
                        remap_y_min = remaped_y

                if y == ymax:
                    if remaped_y > remap_y_max:
                        remap_y_max = remaped_y

                # 某一个位置的 y 值应该为哪个位置的 y 值
                img_y[y, x] = remaped_y
                # 某一个位置的 x 值应该为哪个位置的 x 值
                img_x[y, x] = x

        remaped_text_box_pnts = [
            [xmin, remap_y_min],
            [xmax, remap_y_min],
            [xmax, remap_y_max],
            [xmin, remap_y_max]
        ]

        # TODO: use cuda::remap
        dst = cv2.remap(word_img, img_x, img_y, cv2.INTER_CUBIC)
        return dst, remaped_text_box_pnts

    def _remap_y(self, x, max_val):
        return int(max_val * np.math.sin(2 * 3.14 * x / self.cfg.curve.period))
