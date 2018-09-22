import argparse
import os
import random
import sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../', '../')))
from libs.utils import prob
from textrenderer.corpus.chn_corpus import ChnCorpus


class BalanceCorpus(ChnCorpus):
    # higher -> slower
    BALANCE_TIMES = 1000

    # 认为是低频词
    LESS_CHAR_FACTOR = 0.1

    # 认为是高频词
    MOST_CHAR_FACTOR = 3

    # higher -> more balance?
    LESS_CHAR_FRACTION = 1

    def __init__(self, args):
        super().__init__(args.chars_file, args.corpus_dir, args.length)

        self.args = args
        self.num_balance = args.num_img // (BalanceCorpus.BALANCE_TIMES + 1)

        self.labels = []
        self.chars_avg_count = 0
        self.chars_count_dict = {}
        self.less_chars_count_thres = 0
        self.less_chars_index = {}
        self.less_chars_index_keys = []
        self.corpus = ''.join(self.corpus)

        print("Average char count for all images: %d" % (args.num_img * args.length // len(self.charsets)))

    def run(self):
        """
        Generate a labels.txt file
        """
        i = 1
        while i <= (args.num_img + BalanceCorpus.BALANCE_TIMES):
            if i % self.num_balance == 0:
                print("%d/%d Check char frequency..." % (i, self.args.num_img))
                self.count_char_freq()
                print("Average char count: %d" % self.chars_avg_count)
                print("Low frequency chars threshold: %d" % self.less_chars_count_thres)
                print("Low frequency chars count: %d" % len(self.less_chars_index_keys))
                i += 1

            label = self.get_sample()

            if self.freq_check(label):
                self.labels.append(label)
                i += 1
            else:
                continue

            if i % 100 == 0:
                print("%d/%d" % (i, self.args.num_img), end='\r')

        with open(self.args.output_file, mode='w', encoding='utf-8') as f:
            for t in self.labels:
                f.write(t + '\n')

    def get_sample(self):
        if self.char_freq_counted():
            if prob(BalanceCorpus.LESS_CHAR_FRACTION):
                # 找到低频词所在的位置，从这个位置截取文字段
                key = random.choice(self.less_chars_index_keys)
                index = random.choice(self.less_chars_index[key])

                num_prefix_chars = random.choice(range(0, self.length))

                start = max(index - num_prefix_chars, 0)
            else:
                start = np.random.randint(0, len(self.corpus) - self.length)
        else:
            start = np.random.randint(0, len(self.corpus) - self.length)

        word = self.corpus[start:start + self.length]

        return word

    def char_freq_counted(self):
        return self.chars_avg_count != 0

    def freq_check(self, text: str) -> bool:
        if not self.char_freq_counted():
            return True

        # total = 0
        # for c in text:
        #     # print("%s, %d" % (c, self.char_count_dict.get(c, 0)))
        #     total += self.chars_count_dict.get(c, 0)
        # mean = total // len(text)

        max_c = max(text, key=lambda x: self.chars_count_dict.get(x, 0))
        max_count = self.chars_count_dict.get(max_c, 0)

        # min_c = min(text, key=lambda x: self.chars_count_dict.get(x, 0))
        # min_count = self.chars_count_dict.get(min_c, 0)

        if max_count > self.chars_avg_count * BalanceCorpus.MOST_CHAR_FACTOR:
            # print("mean: %d, avg: %d" % (mean, self.char_avg_count * Balancer.FACTOR))
            return False

        # if min_count < self.char_avg_count:
        #     return True

        return True

    def count_char_freq(self):
        """
        统计已经生成的 labels 中的字符频率
        - 计算每个字符数量
        - 计算字符平均数量
        - 保存当前的低频词在语料中的索引
        """
        data = ''.join(self.labels)
        total_chars_count = len(data)

        chars_count_dict = defaultdict(int)
        for c in data:
            chars_count_dict[c] += 1

        self.chars_avg_count = total_chars_count // len(list(chars_count_dict.keys()))
        self.less_chars_count_thres = self.chars_avg_count * BalanceCorpus.LESS_CHAR_FACTOR

        less_char_index = defaultdict(list)

        for i, c in enumerate(self.corpus):
            if i > len(self.corpus) - self.length:
                break

            count = chars_count_dict.get(c, 0)
            if count < self.less_chars_count_thres:
                less_char_index[c].append(i)

        self.chars_count_dict = chars_count_dict
        self.less_chars_index = less_char_index
        self.less_chars_index_keys = list(less_char_index.keys())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chars_file', type=str, default='./data/chars/chn.txt')
    parser.add_argument('--corpus_dir', type=str, default='/home/cwq/data/ocr/corpus/chn')
    parser.add_argument('--output_file', type=str, default='./output2/labels.txt')
    parser.add_argument('--length', type=int, default=10)
    parser.add_argument('--num_img', type=int, default=100000)

    args = parser.parse_args()

    out_dir = os.path.abspath(os.path.dirname(args.output_file))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return args


if __name__ == "__main__":
    args = parse_args()
    balancer = BalanceCorpus(args)
    balancer.run()
