import random
import numpy as np
import pickle
import os
from textrenderer.corpus.corpus import Corpus


class ChnCorpus(Corpus):
    def load(self):
        """
        Load one corpus file as one line , and get random {self.length} words as result
        """
        self.load_corpus_path()

        for i, p in enumerate(self.corpus_path):
            print_end = '\n' if i == len(self.corpus_path) - 1 else '\r'
            print("Loading chn corpus: {}/{}".format(i + 1, len(self.corpus_path)), end=print_end)
            if os.path.exists(p[:-3]+"cache"):
                print("Loading from cache:{}".format(p[:-3]+"cache"))
                with open(p[:-3]+"cache", 'rb') as f:
                    whole_line = pickle.load(f)["cache"]
            else:
                print("Loading from txt:{}".format(p))
                with open(p, encoding='utf-8') as f:
                    data = f.readlines()

                lines = []
                for line in data:
                    line_striped = line.strip()
                    line_striped = line_striped.replace('\u3000', '')  # blank
                    line_striped = line_striped.replace('\u00A0', '')  # blank
                    line_striped = line_striped.replace('\u0020', '')  # blank
                    line_striped = line_striped.replace('&nbsp', '')  # blank
                    line_striped = line_striped.replace("\00", "")

                    if line_striped != u'' and len(line.strip()) > 1:
                        lines.append(line_striped)

                # 所有行合并成一行
                split_chars = ['、', '.', ',', '，', '：', '-', ' ', ';', '。']
                splitchar = random.choice(split_chars)
                whole_line = splitchar.join(lines)  # 同一文档都用一个random出来的符号链接

                # 在 crnn/libs/label_converter 中 encode 时还会进行过滤
                # only keep words in chn.txt
                whole_line = ''.join(filter(lambda x: x in self.charsets, whole_line))
                tmp = dict()
                tmp["cache"] = whole_line
                with open(p[:-3]+"cache", 'wb') as f:
                    print("Building cache:{}".format(p[:-3] + "cache"))
                    pickle.dump(tmp, f, pickle.HIGHEST_PROTOCOL)

            # only keep length enough corpus
            if len(whole_line) > self.length:
                self.remap.append(self.remap_dataset(p, whole_line))
                self.corpus.append(whole_line)

    def get_sample(self, img_index):
        # 每次 gen_word，随机选一个语料文件，随机获得长度为 word_length 的字符
        # randomly select a corpus and get word_length chars when gen_word
        line = random.choice(self.corpus)

        start = np.random.randint(0, len(line) - self.length)

        word = line[start:start + self.length]
        return word

    def select_sample(self, special_word):
        # random select a corpus
        idx = random.randint(0, len(self.remap)-1)
        # random choice a special word position
        if special_word not in self.remap[idx].keys():
            catch = random.choice(self.remap[idx][special_word])
        else:
            remap_list_id = list(range(len(self.remap)))  # must use list,py3 default is range type,py2 is list
            random.shuffle(remap_list_id)
            for i in remap_list_id:
                if special_word in self.remap[i].keys():
                    idx = i
                    catch = random.choice(self.remap[idx][special_word])
                    break
                else:
                    print("cannot find words {} in corpus".format(special_word))
                    catch = -1
                    continue
        # catch = -1  # test branch
        if catch >= 0:
            max_length = len(self.corpus[idx])
            offset = random.randint(catch-self.length, catch)
            if offset+self.length <= max_length:
                word = self.corpus[idx][offset:offset+self.length]
            # elif offset+self.length > max_length:
            else:
                if max_length <= self.length:
                    word = self.corpus[idx]  # may cause shorter sample img.you can change it by yourself
                else:
                    word = self.corpus[idx][-self.length:]
        else:
            print("Cannot find words{} in all corpus,using random select and insert it".format(special_word))
            word = self.get_sample("")
            insert = random.randint(1, len(word)-1)
            pre_word = word[:insert - 1]
            las_word = word[insert:]
            word = "".join((pre_word, special_word, las_word))

        return word

    @staticmethod
    def remap_dataset(p, whole_line):
        if os.path.exists(p[:-3]+"remap.cache"):
            print("Loading from remap cache:{}".format(p[:-3] + "remap.cache"))
            with open(p[:-3] + "remap.cache", 'rb') as f:
                remap = pickle.load(f)
            return remap
        remap = dict()
        for i, word in enumerate(whole_line):
            if word in remap.keys():
                remap[word].append(i)
            else:
                remap[word] = [i]
        with open(p[:-3]+"remap.cache", "wb") as f:
            print("Building remap cache:{}".format(p[:-3] + "remap.cache"))
            pickle.dump(remap, f, pickle.HIGHEST_PROTOCOL)
        return remap
