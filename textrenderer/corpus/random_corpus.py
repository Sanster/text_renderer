import random

from textrenderer.corpus.corpus import Corpus


class RandomCorpus(Corpus):
    """
    Load charsets and generate random word line from charsets
    """

    def load(self):
        pass

    def get_sample(self, img_index):
        word = ''
        for _ in range(self.length):
            word += random.choice(self.charsets)
        return word

