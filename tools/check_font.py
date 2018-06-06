import argparse
from itertools import chain
import glob
import sys
import os

from fontTools.ttLib import TTFont, TTCollection
from fontTools.unicode import Unicode

sys.path.append('./')
from libs.utils import load_chars


def check_font_charset(ttf, charset):
    chars = chain.from_iterable([y + (Unicode[y[0]],) for y in x.cmap.items()] for x in ttf["cmap"].tables)

    chars_int = []
    for c in chars:
        chars_int.append(c[0])

    not_support_count = 0
    not_support_chars = []
    for c in charset:
        if ord(c) not in chars_int:
            not_support_count += 1
            not_support_chars.append(c)

    ttf.close()
    return not_support_chars


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
        if p.endswith('ttc'):
            ttc = TTCollection(p)
            for f in ttc.fonts:
                fonts["%s_1" % p] = f

        if p.endswith('ttf') or p.endswith('TTF') or p.endswith('otf'):
            ttf = TTFont(p, 0, allowVID=0,
                         ignoreDecompileErrors=True,
                         fontNumber=-1)

            fonts[p] = ttf

    useful_fonts = []
    for k, v in fonts.items():
        print("checking font %s" % k)
        not_support_chars = check_font_charset(v, charset)

        if len(not_support_chars) != 0:
            print("chars not supported(%d): " % len(not_support_chars))
            print(not_support_chars)
            if args.delete:
                os.remove(k)
        else:
            useful_fonts.append(k)


    print("%d fonts support all chars(%d) in %s:" % (len(useful_fonts), len(charset), args.chars_file))
    print(useful_fonts)
