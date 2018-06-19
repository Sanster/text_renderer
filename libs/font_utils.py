import os
import pickle
import glob
from itertools import chain

from fontTools.ttLib import TTCollection, TTFont
from fontTools.unicode import Unicode

from .utils import md5, load_chars


def get_font_paths(fonts_dir):
    """
    :param fonts_dir: folder contains ttf、otf or ttc format font
    :return: path of all fonts
    """
    print('Load fonts from %s' % os.path.abspath(fonts_dir))
    fonts = glob.glob(os.path.join(fonts_dir, "*.*"))
    print("Total fonts num: %d" % len(fonts))

    if len(fonts) == 0:
        print("Not found fonts in fonts_dir")
        exit(-1)
    return fonts


def load_font(font_path):
    """
    Read ttc, ttf, otf font file, return a TTFont object
    """

    # ttc is collection of ttf
    if font_path.endswith('ttc'):
        ttc = TTCollection(font_path)
        # assume all ttfs in ttc file have same supported chars
        return ttc.fonts[0]

    if font_path.endswith('ttf') or font_path.endswith('TTF') or font_path.endswith('otf'):
        ttf = TTFont(font_path, 0, allowVID=0,
                     ignoreDecompileErrors=True,
                     fontNumber=-1)

        return ttf


def check_font_chars(ttf, charset):
    """
    Get font supported chars and unsupported chars
    :param ttf: TTFont ojbect
    :param charset: chars
    :return: unsupported_chars, supported_chars
    """
    chars = chain.from_iterable([y + (Unicode[y[0]],) for y in x.cmap.items()] for x in ttf["cmap"].tables)

    chars_int = []
    for c in chars:
        chars_int.append(c[0])

    unsupported_chars = []
    supported_chars = []
    for c in charset:
        if ord(c) not in chars_int:
            unsupported_chars.append(c)
        else:
            supported_chars.append(c)

    ttf.close()
    return unsupported_chars, supported_chars


def get_fonts_chars(fonts, chars_file):
    """
    loads/saves font supported chars from cache file
    :param fonts: list of font path. e.g ['./data/fonts/msyh.ttc']
    :param chars_file: arg from parse_args
    :return: dict, key -> font_path, value -> font supported chars
    """
    out = {}

    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', '.caches'))
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    chars = load_chars(chars_file)
    chars = ''.join(chars)

    for font_path in fonts:
        string = ''.join([font_path, chars])
        file_md5 = md5(string)

        cache_file_path = os.path.join(cache_dir, file_md5)

        if not os.path.exists(cache_file_path):
            ttf = load_font(font_path)
            _, supported_chars = check_font_chars(ttf, chars)
            print('Save font(%s) supported chars(%d) to cache' % (font_path, len(supported_chars)))

            with open(cache_file_path, 'wb') as f:
                pickle.dump(supported_chars, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(cache_file_path, 'rb') as f:
                supported_chars = pickle.load(f)
            print('Load font(%s) supported chars(%d) from cache' % (font_path, len(supported_chars)))

        out[font_path] = supported_chars

    return out


if __name__ == '__main__':
    font_paths = get_font_paths('./data/fonts/chn')
    char_file = './data/chars/chn.txt'
    chars = get_fonts_chars(font_paths, char_file)
    print(chars)
