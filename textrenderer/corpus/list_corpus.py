from textrenderer.corpus.corpus import Corpus
import numpy as np


class ListCorpus(Corpus):
    """
    get_sample from corpus line by line
    """

    def load(self):
        self.load_corpus_path()

        for i, p in enumerate(self.corpus_path):
            print("Load corpus: %s" % p)
            with open(p, encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                self.corpus.append(line.strip())

        print("Total lines: {}".format(len(self.corpus)))

    def get_sample(self, img_index):
        index = img_index % len(self.corpus)
        return self.corpus[index]
