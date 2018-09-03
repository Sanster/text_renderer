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
        h = word_img.shape[0]
        w = word_img.shape[1]

        img_x = np.zeros((h, w), np.float32)
        img_y = np.zeros((h, w), np.float32)

        xmin = text_box_pnts[0][0]
        xmax = text_box_pnts[1][0]
        ymin = text_box_pnts[0][1]
        ymax = text_box_pnts[2][1]

        for y in range(h):
            for x in range(w):
                # 某一个位置的 y 值应该为哪个位置的 y 值
                img_y[y, x] = y + self._remap_y(x)
                # 某一个位置的 x 值应该为哪个位置的 x 值
                img_x[y, x] = x

        # TODO: put find min/max in for loop
        remap_y_offset_min_x = min(list(range(xmin, xmax)), key=lambda x: self._remap_y(x))
        remap_y_offset_max_x = max(list(range(xmin, xmax)), key=lambda x: self._remap_y(x))
        remap_y_offset_min = self._remap_y(remap_y_offset_min_x)
        remap_y_offset_max = self._remap_y(remap_y_offset_max_x)

        remaped_text_box_pnts = [
            [xmin, ymin + remap_y_offset_min],
            [xmax, ymin + remap_y_offset_min],
            [xmax, ymax + remap_y_offset_max],
            [xmin, ymax + remap_y_offset_max]
        ]

        # TODO: use cuda::remap
        dst = cv2.remap(word_img, img_x, img_y, cv2.INTER_CUBIC)
        return dst, remaped_text_box_pnts

    def _remap_y(self, x):
        return int(self.cfg.curve.max * np.math.sin(2 * 3.14 * x / self.cfg.curve.period))
