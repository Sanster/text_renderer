#!/usr/env/bin python3
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_img', type=int, default=20, help="Number of images to generate")

    parser.add_argument('--length', type=int, default=10,
                        help='Chars(chn) or words(eng) in a image. For eng corpus mode, default length is 3')

    parser.add_argument('--clip_max_chars', action='store_true', default=False,
                        help='For training a CRNN model, max number of chars in an image'
                             'should less then the width of last CNN layer.')

    parser.add_argument('--img_height', type=int, default=32)
    parser.add_argument('--img_width', type=int, default=256,
                        help="If 0, output images will have different width")

    parser.add_argument('--chars_file', type=str, default='./data/chars/chn.txt',
                        help='Chars allowed to be appear in generated images.')

    parser.add_argument('--config_file', type=str, default='./configs/default.yaml',
                        help='Set the parameters when rendering images')

    parser.add_argument('--fonts_list', type=str, default='./data/fonts_list/chn.txt',
                        help='Fonts file path to use')

    parser.add_argument('--bg_dir', type=str, default='./data/bg',
                        help="Some text images(according to your config in yaml file) will"
                             "use pictures in this folder as background")

    parser.add_argument('--corpus_dir', type=str, default="./data/corpus",
                        help='When corpus_mode is chn or eng, text on image will randomly selected from corpus.'
                             'Recursively find all txt file in corpus_dir')

    parser.add_argument('--corpus_mode', type=str, default='chn', choices=['random', 'chn', 'eng', 'list'],
                        help='Different corpus type have different load/get_sample method'
                             'random: random pick chars from chars file'
                             'chn: pick continuous chars from corpus'
                             'eng: pick continuous words from corpus, space is included in label')

    parser.add_argument('--output_dir', type=str, default='./output', help='Images save dir')

    parser.add_argument('--tag', type=str, default='default', help='output images are saved under output_dir/{tag} dir')

    parser.add_argument('--debug', action='store_true', default=False, help="output uncroped image")

    parser.add_argument('--viz', action='store_true', default=False)

    parser.add_argument('--strict', action='store_true', default=False,
                        help="check font supported chars when generating images")

    parser.add_argument('--gpu', action='store_true', default=False, help="use CUDA to generate image")

    parser.add_argument('--num_processes', type=int, default=None,
                        help="Number of processes to generate image. If None, use all cpu cores")

    flags, _ = parser.parse_known_args()
    flags.save_dir = os.path.join(flags.output_dir, flags.tag)

    if os.path.exists(flags.bg_dir):
        num_bg = len(os.listdir(flags.bg_dir))
        flags.num_bg = num_bg

    if not os.path.exists(flags.save_dir):
        os.makedirs(flags.save_dir)

    if flags.num_processes == 1:
        parser.error("num_processes min value is 2")

    return flags


if __name__ == '__main__':
    args = parse_args()
    print(args.corpus_dir)
