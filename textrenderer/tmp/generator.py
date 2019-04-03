#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random

import cv2
import utils
import config
import argparse
import charset

from multiprocessing import Pool


def gen_image(fonts, text, bgs, smudginess):
    bg = utils.load_bg(random.choice(bgs))

    smu = None
    if random.randint(0, 100) < config.smudginess_percent:
        smu = random.choice(smudginess)
    
        
    gauss = 0.0
    if random.randint(0, 100) < config.gauss_percent:
    #    gauss = random.randint(1, 3)
        gauss = random.uniform(config.gauss_range[0],config.gauss_range[1])

    underline = False
    if random.randint(0, 100) < config.underline_percent:
        underline = True

    distort_param = None
    if random.randint(0, 100) < config.distort_percent:
        distort_index = random.randint(0, len(config.distortParamList) - 1)
        distort_param = config.distortParamList[distort_index]

    geometry_param = None
    if random.randint(0, 100) < config.geometry_percent:
        geometry_index = random.randint(0, len(config.geometryParamlist) - 1)
        geometry_param = config.geometryParamlist[geometry_index]

    skewing = None
    if random.randint(0, 100) < config.skewing_percent:
        skewing = config.skewing_range

    font_gray = random.randint(config.font_gray[0],config.font_gray[1])
    return utils.gen_line(bg, fonts,font_gray, text,
                        underline=underline,
                        gauss=gauss,
                        distort_param=distort_param,
                        geometry_param=geometry_param,
                        smudginess=smu,
                        skewing=skewing)


def gen_text(idx, text, font_files, bgs, smudginess, out_dir):
    print("gen_text -->idx: ", idx, "text: " + text)
    font_file = random.choice(font_files)
    print('font file -->',font_file)
    fonts = utils.load_fonts_size(font_file, (32, 48))

    if len(text) < 7:
        if random.randint(0, 100) < config.add_biaodian_percent:
            if random.randint(0, 1) == 0:
                text += random.choice(charset.biaodian)
            else:
                text = random.choice(charset.biaodian) + text

    file_name = out_dir + "/%09d" % idx
    im = gen_image(fonts, text, bgs, smudginess)
    if im is not None:
        cv2.imwrite(file_name + ".png", im)
        print("write file to ", file_name+'.png')
        with open(file_name + ".gt.txt", "w") as f:
            f.write(text + "\n")

    return 216


def random_add_char(lines, percent, chset):

    if percent > 0:
        ret = []
        for line in lines:
            if random.randint(0, 100) < percent:
                pos = random.randint(0, len(line) - 2)
                ret.append(line[:pos] + random.choice(chset) + line[pos:])
        return ret
    else:
        return lines


def gen_chinese(out_dir, count, worker_num):

    lines = []
    for i in range(0, count):
        lines.extend(utils.get_dic_lines(charset.charset))
    lines = random_add_char(lines, config.add_biaodian_percent, charset.biaodian)

    gen_lines(lines, out_dir, worker_num)


def gen_lines(lines, out_dir, worker_num):
    fonts = utils.load_fonts_path(config.fonts)
    smudginess = utils.load_smudginess(config.smuDir)
    bgs = utils.load_bgs(config.bgDir)
    l = len(lines)
    step = 10000
    count = int(l / step)
    for i in range(0, count):
        gen_line_range(out_dir, worker_num, lines, i * 10000, (i + 1) * 10000, fonts, bgs, smudginess)
    gen_line_range(out_dir, worker_num, lines, count * 10000, l, fonts, bgs, smudginess)
    print('job done!!!!')


def gen_line_range(out_dir, worker_num, lines, star, end, fonts, bgs, smudginess):

    for idx in range(star, end):
        out_path = out_dir + '\\' + str(int(idx / 1000)) + '\\'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

    pool = Pool(worker_num)
    result = []
    for idx in range(star, end):
        out_path = out_dir + '/' + str(int(idx / 1000)) + '/'
        result.append(pool.apply_async(gen_text, (idx, lines[idx], fonts, bgs, smudginess, out_path)))
    pool.close()
    pool.join()
    print("wait process to finish")
    return [res.get() for res in result]

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Generate text line training data")
    parser.add_argument('-o', '--base', default=r'E:\deeplearn\OCR\Sample\trainnew', help='output directory, default: %(default)s')
    parser.add_argument('-c', '--count', default=1, type=int)

    parser.add_argument('-w', '--worknum', default=8, type=int)

    args = parser.parse_args()

    gen_chinese(args.base + "/chinese/", args.count, args.worknum)







