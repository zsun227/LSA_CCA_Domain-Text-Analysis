#!/usr/bin/env python
# coding: utf-8

# In[10]:

import numpy as np
import os
import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import datetime
# import tensorflow as tf
# from tensorflow.contrib.tensorboard.plugins import projector

# print('gensim Version: %s' % (gensim.__version__))

class WordEmbedding:
    __author__ = "Edward Ma"
    __copyright__ = "Copyright 2018, Edward Ma"
    __credits__ = ["Edward Ma"]
    __license__ = "Apache"
    __version__ = "2.0"
    __maintainer__ = "Edward Ma"
    __email__ = "makcedward@gmail.com"

    def __init__(self, verbose=0):
        self.verbose = verbose
        
        self.model = {}
        
    def convert(self, source, input_file_path, output_file_path):
        if source == 'glove':
            input_file = datapath(input_file_path)
            output_file = get_tmpfile(output_file_path)
            glove2word2vec(input_file, output_file)
        elif source == 'word2vec':
            pass
        elif source == 'fasttext':
            pass
        else:
            raise ValueError('Possible value of source are glove, word2vec, fasttext')
        
    def load(self, source, file_path):
        print(datetime.datetime.now(), 'start: loading', source)
        if source == 'glove':
            self.model[source] = gensim.models.KeyedVectors.load_word2vec_format(file_path)
        elif source == 'word2vec':
            self.model[source] = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)
        elif source == 'fasttext':
            self.model[source] = gensim.models.wrappers.FastText.load_fasttext_format(file_path)
        else:
            raise ValueError('Possible value of source are glove, word2vec, fasttext')
            
        print(datetime.datetime.now(), 'end: loading', source)
            
        return self
    
    def get_model(self, source):
        if source not in ['glove', 'word2vec', 'fasttext']:
            raise ValueError('Possible value of source are glove, word2vec, fasttext')
            
        return self.model[source]
    
    def get_words(self, source, size=None):
        if source not in ['glove', 'word2vec', 'fasttext']:
            raise ValueError('Possible value of source are glove, word2vec, fasttext')
        
        if source in ['glove', 'word2vec']:
            if size is None:
                return [w for w in self.get_model(source=source).vocab]
            else:
                results = []
                for i, word in enumerate(self.get_model(source=source).vocab):
                    if i >= size:
                        break
                        
                    results.append(word)
                return results
            
        elif source in ['fasttext']:
            if size is None:
                return [w for w in self.get_model(source=source).wv.vocab]
            else:
                results = []
                for i, word in enumerate(self.get_model(source=source).wv.vocab):
                    if i >= size:
                        break
                        
                    results.append(word)
                return results
        
        return Exception('Unexpected flow')
    
    def get_dimension(self, source):
        if source not in ['glove', 'word2vec', 'fasttext']:
            raise ValueError('Possible value of source are glove, word2vec, fasttext')
        
        if source in ['glove', 'word2vec']:
            return self.get_model(source=source).vectors[0].shape[0]
            
        elif source in ['fasttext']:
            word = self.get_words(source=source, size=1)[0]
            return self.get_model(source=source).wv[word].shape[0]
        
        return Exception('Unexpected flow')
    
    def get_vectors(self, source, words=None):
        if source not in ['glove', 'word2vec', 'fasttext']:
            raise ValueError('Possible value of source are glove, word2vec, fasttext')
        
        if source in ['glove', 'word2vec', 'fasttext']:
            if words is None:
                words = self.get_words(source=source)
            
            embedding = np.empty((len(words), self.get_dimension(source=source)), dtype=np.float32)            
            for i, word in enumerate(words):
                embedding[i] = self.get_vector(source=source, word=word)
                
            return embedding
        
        return Exception('Unexpected flow')

    def get_vector(self, source, word, oov=None):
        if source not in ['glove', 'word2vec', 'fasttext']:
            raise ValueError('Possible value of source are glove, word2vec, fasttext')
            
        if source not in self.model:
            raise ValueError('Did not load %s model yet' % source)
        
        try:
            return self.model[source][word]
        except KeyError as e:
            raise KeyError('word is not in embedding')


