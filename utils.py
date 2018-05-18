import numpy as np
import os
import pickle
import fastText as ft




DATA_PATH = 'data/'

def save_array(dat, filename):
    np.save(os.path.join(DATA_PATH, filename), dat) 
    
def load_array(filename):
    return np.load(os.path.join(DATA_PATH, filename)) 


def get_vecs(ft_vecs):
    vecd = {w:ft_vecs.get_word_vector(w) for w in ft_vecs.get_words()}
    pickle.dump(vecd, open(word_vector_path))
    return vecd


    