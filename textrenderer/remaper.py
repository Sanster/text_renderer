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
        pass
