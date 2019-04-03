import random
import numpy as np
import h5py
import glob
import os

from textrenderer.corpus.corpus import Corpus


class npyChnCorpus(Corpus):
    """
    hdf5 file format 
    """
    def load(self):
        npyfiles = glob.glob(self.corpus_dir+'/**.npy')
        print('find corpus:',npyfiles)
        for npyfile in npyfiles:
            self.corpus += np.load(npyfile).tolist()
        self.lines = len(self.corpus)
        self.index = 0 

    def get_sample(self, img_index):
        
        index = np.random.randint(0, self.lines)
        word = ''.join(self.corpus[index])
        return word
