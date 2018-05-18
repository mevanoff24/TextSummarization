import numpy as np
from more_itertools import chunked
from math import ceil
from itertools import chain
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
import re
import os
import pickle
import fastText as ft

from keras.preprocessing.sequence import pad_sequences

import spacy
from spacy.symbols import ORTH


def flattenlist(listoflists):
    return list(chain.from_iterable(listoflists))

class Tokenizer():
    def __init__(self, lang='en'):
        self.tokenizer = spacy.load(lang)
        self.re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
        for w in ('_eos_','_bos_','_unk_'):
            self.tokenizer.tokenizer.add_special_case(w, [{ORTH: w}])
            
    def sub_br(self, X): 
        return self.re_br.sub("\n", X)
    
    def spacy_tokenizer(self, X):
        return [x.text for x in self.tokenizer.tokenizer(self.sub_br(X.lower())) if not x.is_space]
    
    def proc_text(self, X):
        s = re.sub(r'([/#])', r' \1 ', X)
        s = re.sub(' {2,}', ' ', X)
        return self.spacy_tokenizer(s)
       
    @staticmethod
    def proc_all(X):
        tok = Tokenizer()
        return [tok.proc_text(x) for x in X]

    def fit_transform(self, X):
        core_usage = (cpu_count() + 1) // 2
        with Pool(core_usage) as p:
            chunk_size = ceil(len(X) / core_usage)
            results = p.map(Tokenizer.proc_all, chunked(X, chunk_size), chunksize=1)
        return flattenlist(results)
    
    def transform(self, X):
        return Tokenizer.proc_all(X)
    
    
    


class TextDataSet(object):
    def __init__(self, max_vocab, maxlen, min_freq=1, padding='pre'):
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.maxlen = maxlen
        self.padding = padding
        self.tokenizer = Tokenizer()
    
    def fit(self, text, tokenize=True):
        if tokenize:
            text = self.tokenizer.fit_transform(text)
        self.freq = Counter(p for sent in text for p in sent)
        self.idx2word = [word for word, count in self.freq.most_common(self.max_vocab) if count > self.min_freq]
        self.idx2word.insert(0, '_unk_')
        self.idx2word.insert(1, '_pad_')
        self.idx2word.insert(2, '_bos_')
        self.idx2word.insert(3, '_eos_')
        self.word2idx = defaultdict(lambda: 0, {word: i for i, word in enumerate(self.idx2word)})
        self.pad_int = self.word2idx['_pad_']
        return text
        
    def fit_transform(self, text, tokenize=True):
        text = self.fit(text, tokenize=tokenize)
        text_padded = self.internal_transform(text, tokenize=False)
        return np.array(text_padded)
    
    def internal_transform(self, text, tokenize=True):
        if tokenize:       
            text = self.tokenizer.fit_transform(text)
        text_ints = np.array([[self.word2idx[i] for i in sent] for sent in text])
        text_padded = pad_sequences(text_ints, maxlen=self.maxlen, padding=self.padding, value=self.pad_int)
        return np.array(text_padded)
    
    def transform(self, text, tokenize=True, word2idx=None, maxlen=None, padding=None):
        if tokenize:       
            text = self.tokenizer.fit_transform(text)
        if word2idx:
            self.word2idx = word2idx
        if maxlen:
            self.maxlen = maxlen
        if padding:
            self.padding = padding
        text_ints = np.array([[self.word2idx[i] for i in sent] for sent in text])
        text_padded = pad_sequences(text_ints, maxlen=self.maxlen, padding=self.padding, value=self.pad_int)
        return np.array(text_padded)
    
    
def limit_unk_vocab(train_text, train_summary, all_text_model, max_unk_text=1, max_unk_summary=1):
    train_text_reduced = []
    train_summary_reduced = []

    for txt, sumy in zip(train_text, train_summary):
        unk_txt = len([x for x in txt if x == all_text_model.word2idx['_unk_']])
        unk_sumy = len([x for x in sumy if x == all_text_model.word2idx['_unk_']])
        if (unk_txt <= max_unk_text) and (unk_sumy <= max_unk_summary):
            train_text_reduced.append(txt.tolist())
            train_summary_reduced.append(sumy.tolist())
        
    assert(len(train_text_reduced) == len(train_summary_reduced))
    print('new text size', len(train_text_reduced))
    print('new summary size', len(train_summary_reduced))

    return np.array(train_text_reduced), np.array(train_summary_reduced)

