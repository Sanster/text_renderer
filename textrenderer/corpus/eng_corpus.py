from textrenderer.corpus.corpus import Corpus
import numpy as np


class EngCorpus(Corpus):
    """
    Load English corpus by words, and get random {self.length} words as result
    """

    def load(self):
        self.load_corpus_path()

        for i, p in enumerate(self.corpus_path):
            print("Load {} th eng corpus".format(i))
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

    def get_sample(self, img_index):
        start = np.random.randint(0, len(self.corpus) - self.length)
        words = self.corpus[start:start + self.length]
        word = ' '.join(words)
        return word
