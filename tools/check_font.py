import argparse
import glob
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../', '../')))
from libs.utils import load_chars
from libs.font_utils import check_font_chars, load_font

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find chars not support by some fonts')
    parser.add_argument('--chars_file', type=str, default='./data/chars/chn.txt')
    parser.add_argument('--font_dir', type=str, default='./data/fonts/eng')
    parser.add_argument('--delete', action="store_true", default=False,
                        help='whether or not to delete font which not full support the chars_file')

    args, _ = parser.parse_known_args()

    charset = load_chars(args.chars_file)
    font_paths = glob.glob(args.font_dir + '/*.*')

    fonts = {}
    for p in font_paths:
        ttf = load_font(p)
        fonts[p] = ttf

    useful_fonts = []
    for k, v in fonts.items():
        unsupported_chars, _ = check_font_chars(v, charset)

        print("font: %s ,chars unsupported: %d" % (k, len(unsupported_chars)))
        if len(unsupported_chars) != 0:
            if args.delete:
                os.remove(k)
        else:
            useful_fonts.append(k)

    print("%d fonts support all chars(%d) in %s:" % (len(useful_fonts), len(charset), args.chars_file))
    print(useful_fonts)
