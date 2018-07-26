import argparse
import os
import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def analyze_labels(path: str):
    char_count_dict = defaultdict(int)

    with open(path, mode='r', encoding='utf-8') as f:
        data = f.readlines()
        # 移除换行符、首尾空格
        data = map(lambda l: l[:-1].strip(), data)
        data = ''.join(data)
        total_chars_count = len(data)

        for c in data:
            char_count_dict[c] += 1

    return char_count_dict, total_chars_count


def print_info(chars_count_list, total_chars_count, name, max_count=15):
    print("Info for %s" % name)
    print("Total chars count %d" % total_chars_count)
    freqs = list(map(lambda x: x[1] / total_chars_count, chars_count_list))
    avg_freq = np.mean(freqs)
    std = np.std(freqs)
    print("Average frequence: %f +- %f %%" % (avg_freq, std))

    above_avg_freq = sum(map(lambda x: x > avg_freq, freqs))
    print("Chars freq in training label above average: %d" % above_avg_freq)

    print("Top %d" % max_count)
    count = 0
    for index, (k, v) in enumerate(chars_count_list):
        print("%s %f%% %d" % (k, v / total_chars_count, chars_count_list[index][1]))
        count += 1
        if count > max_count:
            break

    print("Bottom %d" % max_count)
    count = 0
    reversed_list = list(reversed(chars_count_list))
    for index, (k, v) in enumerate(reversed_list):
        print("%s %f%% %d" % (k, v / total_chars_count, reversed_list[index][1]))
        count += 1
        if count > max_count:
            break

    return avg_freq, above_avg_freq


def show_plot(log=False):
    if log:
        plt.yscale('log', nonposy='clip')
    plt.ylabel('Count')
    plt.legend()
    plt.show()


def process_dir(label_dir, log=False):
    label_paths = glob.glob(label_dir + '/*.txt')

    for p in label_paths:
        name = p.split('/')[-1].split('.')[0]
        chars_count_list, total_chars_count = analyze_labels(p)
        print_info(chars_count_list, total_chars_count, name)

        y = list(map(lambda x: x[1], chars_count_list))
        plt.plot(y, label=name)

    show_plot(log)


def process_file(label_file: str, log=False):
    name = "label"
    chars_count_dict, total_chars_count = analyze_labels(label_file)

    # 降序
    chars_count_list = list(sorted(chars_count_dict.items(), key=lambda x: x[1], reverse=True))

    print_info(chars_count_list, total_chars_count, name)

    y = list(map(lambda x: x[1], chars_count_list))

    plt.plot(y, label=name)
    show_plot(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='输入 labels.txt 标签数据，输出标签中每个字符的频率图')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--log', action='store_true', default=False)
    flags, _ = parser.parse_known_args()

    if os.path.isdir(flags.label):
        process_dir(flags.label, flags.log)
    elif os.path.isfile(flags.label):
        process_file(flags.label, flags.log)
