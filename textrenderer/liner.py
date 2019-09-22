import random
import cv2
import numpy as np


class LineState(object):
    tableline_x_offsets = range(8, 40)
    tableline_y_offsets = range(3, 10)
    tableline_thickness = [1, 2]

    # 0/1/2/3: 仅单边（左上右下）
    # 4/5/6/7: 两边都有线（左上，右上，右下，左下）
    tableline_options = range(0, 8)

    middleline_thickness = [1, 2, 3]
    middleline_thickness_p = [0.2, 0.7, 0.1]


class Liner(object):
    def __init__(self, cfg):
        self.linestate = LineState()
        self.cfg = cfg

    def get_line_color(self):
        p = []
        colors = []
        for k, v in self.cfg.line_color.items():
            if k == 'enable':
                continue
            p.append(v.fraction)
            colors.append(k)

        # pick color by fraction
        color_name = np.random.choice(colors, p=p)
        l_boundary = self.cfg.line_color[color_name].l_boundary
        h_boundary = self.cfg.line_color[color_name].h_boundary
        # random color by low and high RGB boundary
        r = np.random.randint(l_boundary[0], h_boundary[0])
        g = np.random.randint(l_boundary[1], h_boundary[1])
        b = np.random.randint(l_boundary[2], h_boundary[2])
        return b, g, r

    def apply(self, word_img, text_box_pnts, word_color):
        """
        :param word_img:  word image with big background
        :param text_box_pnts: left-top, right-top, right-bottom, left-bottom of text word
        :return:
        """
        line_p = []
        funcs = []

        if self.cfg.line.under_line.enable:
            line_p.append(self.cfg.line.under_line.fraction)
            funcs.append(self.apply_under_line)

        if self.cfg.line.table_line.enable:
            line_p.append(self.cfg.line.table_line.fraction)
            funcs.append(self.apply_table_line)

        if self.cfg.line.middle_line.enable:
            line_p.append(self.cfg.line.middle_line.fraction)
            funcs.append(self.apply_middle_line)

        if len(line_p) == 0:
            return word_img, text_box_pnts

        line_effect_func = np.random.choice(funcs, p=line_p)

        if self.cfg.line_color.enable or self.cfg.font_color.enable:
            line_color = self.get_line_color()
        else:
            line_color = word_color + random.randint(0, 10)

        return line_effect_func(word_img, text_box_pnts, line_color)

    def apply_under_line(self, word_img, text_box_pnts, line_color):
        y_offset = random.choice([0, 1])

        text_box_pnts[2][1] += y_offset
        text_box_pnts[3][1] += y_offset

        dst = cv2.line(word_img,
                       (text_box_pnts[2][0], text_box_pnts[2][1]),
                       (text_box_pnts[3][0], text_box_pnts[3][1]),
                       color=line_color,
                       thickness=1,
                       lineType=cv2.LINE_AA)

        return dst, text_box_pnts

    def apply_table_line(self, word_img, text_box_pnts, line_color):
        """
        共有 8 种可能的画法，横线横穿整张 word_img
        0/1/2/3: 仅单边（左上右下）
        4/5/6/7: 两边都有线（左上，右上，右下，左下）
        """
        dst = word_img
        option = random.choice(self.linestate.tableline_options)
        thickness = random.choice(self.linestate.tableline_thickness)

        top_y_offset = random.choice(self.linestate.tableline_y_offsets)
        bottom_y_offset = random.choice(self.linestate.tableline_y_offsets)
        left_x_offset = random.choice(self.linestate.tableline_x_offsets)
        right_x_offset = random.choice(self.linestate.tableline_x_offsets)

        def is_top():
            return option in [1, 4, 5]

        def is_bottom():
            return option in [3, 6, 7]

        def is_left():
            return option in [0, 4, 7]

        def is_right():
            return option in [2, 5, 6]

        if is_top():
            text_box_pnts[0][1] -= top_y_offset
            text_box_pnts[1][1] -= top_y_offset

        if is_bottom():
            text_box_pnts[2][1] += bottom_y_offset
            text_box_pnts[3][1] += bottom_y_offset

        if is_left():
            text_box_pnts[0][0] -= left_x_offset
            text_box_pnts[3][0] -= left_x_offset

        if is_right():
            text_box_pnts[1][0] += right_x_offset
            text_box_pnts[2][0] += right_x_offset

        if is_bottom():
            dst = cv2.line(dst,
                           (0, text_box_pnts[2][1]),
                           (word_img.shape[1], text_box_pnts[3][1]),
                           color=line_color,
                           thickness=thickness,
                           lineType=cv2.LINE_AA)

        if is_top():
            dst = cv2.line(dst,
                           (0, text_box_pnts[0][1]),
                           (word_img.shape[1], text_box_pnts[1][1]),
                           color=line_color,
                           thickness=thickness,
                           lineType=cv2.LINE_AA)

        if is_left():
            dst = cv2.line(dst,
                           (text_box_pnts[0][0], 0),
                           (text_box_pnts[3][0], word_img.shape[0]),
                           color=line_color,
                           thickness=thickness,
                           lineType=cv2.LINE_AA)

        if is_right():
            dst = cv2.line(dst,
                           (text_box_pnts[1][0], 0),
                           (text_box_pnts[2][0], word_img.shape[0]),
                           color=line_color,
                           thickness=thickness,
                           lineType=cv2.LINE_AA)

        return dst, text_box_pnts

    def apply_middle_line(self, word_img, text_box_pnts, line_color):
        y_center = int((text_box_pnts[0][1] + text_box_pnts[3][1]) / 2)

        thickness = np.random.choice(self.linestate.middleline_thickness, p=self.linestate.middleline_thickness_p)

        dst = cv2.line(word_img,
                       (text_box_pnts[0][0], y_center),
                       (text_box_pnts[1][0], y_center),
                       color=line_color,
                       thickness=thickness,
                       lineType=cv2.LINE_AA)

        return dst, text_box_pnts
