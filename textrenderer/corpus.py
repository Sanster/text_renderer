from abc import abstractmethod
import numpy as np
import random
import glob

from libs.utils import prob, load_chars


class Corpus(object):
    def __init__(self, chars_file, corpus_dir=None, length=None):
        self.corpus_dir = corpus_dir
        self.length = length
        self.corpus = []

        self.chars_file = chars_file
        self.charsets = load_chars(chars_file)

        if not isinstance(self, RandomCorpus):
            print("Loading corpus from: " + self.corpus_dir)
            self.corpus_path = glob.glob(self.corpus_dir + '/**/*.txt', recursive=True)
            if len(self.corpus_path) == 0:
                print("Corpus not found.")
                exit(-1)

        self.load()

    @abstractmethod
    def load(self):
        """
        Read corpus from disk to memory
        """
        pass

    @abstractmethod
    def get_sample(self):
        """
        Get word line from corpus in memory
        :return: string
        """
        pass


class RandomCorpus(Corpus):
    """
    Load charsets and generate random word line from charsets
    """

    def load(self):
        pass

    def get_sample(self):
        word = ''
        for _ in range(self.length):
            word += random.choice(self.charsets)
        return word


class EngCorpus(Corpus):
    def load(self):
        for i, p in enumerate(self.corpus_path):
            print("Load {}th eng corpus".format(i))
            with open(p, encoding='utf-8') as f:
                data = f.read()

            lines = data.split('\n')
            for line in lines:
                for word in line.split(' '):
                    word = word.strip()
                    word = ''.join(filter(lambda x: x in self.charsets, word))

                    if word != u'' and len(word) > 2:
                        self.corpus.append(word)
            print("Word count {}".format(len(self.corpus)))

    def get_sample(self):
        start = np.random.randint(0, len(self.corpus) - self.length)
        words = self.corpus[start:start + self.length]
        word = ' '.join(words)
        return word


class ChnCorpus(Corpus):
    def load(self):
        """
        Load one corpus file as one line
        """
        for i, p in enumerate(self.corpus_path):
            print_end = '\n' if i == len(self.corpus_path) - 1 else '\r'
            print("Loading chn corpus: {}/{}".format(i + 1, len(self.corpus_path)), end=print_end)
            with open(p, encoding='utf-8') as f:
                data = f.readlines()

            lines = []
            for line in data:
                line_striped = line.strip()
                line_striped = line_striped.replace('\u3000', '')
                line_striped = line_striped.replace('&nbsp', '')
                line_striped = line_striped.replace("\00", "")

                if line_striped != u'' and len(line.strip()) > 1:
                    lines.append(line_striped)

            # 所有行合并成一行
            split_chars = [',', '，', '：', '-', ' ', ';', '。']
            splitchar = random.choice(split_chars)
            whole_line = splitchar.join(lines)

            # 在 crnn/libs/label_converter 中 encode 时还会进行过滤
            whole_line = ''.join(filter(lambda x: x in self.charsets, whole_line))

            if len(whole_line) > self.length:
                self.corpus.append(whole_line)

    def get_sample(self):
        # 每次 gen_word，随机选一个预料文件，随机获得长度为 word_length 的字符
        line = random.choice(self.corpus)

        start = np.random.randint(0, len(line) - self.length)

        word = line[start:start + self.length]
        return word
