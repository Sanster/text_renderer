from textrenderer.corpus.chn_corpus import ChnCorpus
from textrenderer.corpus.eng_corpus import EngCorpus
from textrenderer.corpus.random_corpus import RandomCorpus


def corpus_factory(corpus_mode: str, chars_file: str, corpus_dir: str, length: int):
    corpus_classes = {
        "random": RandomCorpus,
        "chn": ChnCorpus,
        "eng": EngCorpus
    }

    corpus_class = corpus_classes[corpus_mode]

    if length == 10 and corpus_mode == 'eng':
        length = 3

    return corpus_class(chars_file=chars_file, corpus_dir=corpus_dir, length=length)
