#!/usr/env/bin python3
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', action='store_true', default=False, help="use CUDA to generate image")
    parser.add_argument('--num_processes', type=int, default=None,
                        help="Number of processes to generate image. If None, use all cpu cores")
    parser.add_argument('--num_img', type=int, default=10, help="Number of images to generate")
    parser.add_argument('--corpus_mode', type=str, default='chn', choices=['random', 'chn', 'eng'],
                        help='Different corpus type have different load/get_sample method')

    parser.add_argument('--length', type=int, default=10,
                        help='Number of chars in a image, only works for chn corpus_mode')
    parser.add_argument('--img_height', type=int, default=32)
    parser.add_argument('--img_width', type=int, default=256)

    parser.add_argument('--debug', action='store_true', default=False, help="output uncroped image")
    parser.add_argument('--viz', action='store_true', default=False)

    parser.add_argument('--chars_file', type=str, default='./data/chars/chn.txt')

    parser.add_argument('--fonts_dir', type=str, default='./data/fonts/')
    parser.add_argument('--bg_dir', type=str, default='./data/bg')
    parser.add_argument('--corpus_dir', type=str, default='./data/corpus')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--tag', type=str, default='default', help='output images are saved under output_dir/{tag} dir')

    parser.add_argument('--line', action='store_true', default=False)
    parser.add_argument('--noise', action='store_true', default=False)

    flags, _ = parser.parse_known_args()
    flags.save_dir = os.path.join(flags.output_dir, flags.tag)

    if os.path.exists(flags.bg_dir):
        num_bg = len(os.listdir(flags.bg_dir))
        flags.num_bg = num_bg

    if not os.path.exists(flags.save_dir):
        os.makedirs(flags.save_dir)

    return flags
