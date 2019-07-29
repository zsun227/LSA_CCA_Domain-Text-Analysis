#!/usr/bin/env python
# coding: utf-8

# In[104]:


import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import regexp
import rccaMod
import numpy as np
import pandas as pd
#import csv
from joblib import dump, load
import sklearn
import nltk
nltk.download('punkt')
import gensim
from nltk import word_tokenize
import re
from string import punctuation
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
from scipy.spatial import distance
from joblib import dump, load


# In[61]:


#  clean raw text

def str_punc(s):
    return ''.join(c for c in s if c not in punctuation)
def processing(text):
    wo = word_tokenize(str_punc(text))
    st = ''.join(c.lower()+' ' for c in wo)
    return st
def cleanText(text):
    # Replace non-ASCII characters with printable ASCII. 
    # Use HTML entities when possible
    if None == text:
        return ''

    
    text = re.sub(r'\x85', '', text) # replace ellipses
    text = re.sub(r'\x91', "‘", text)  # replace left single quote
    text = re.sub(r'\x92', "’", text)  # replace right single quote
    text = re.sub(r'\x93', '“', text)  # replace left double quote
    text = re.sub(r'\x94', '”', text)  # replace right double quote
    text = re.sub(r'\x95', '•', text)   # replace bullet
    text = re.sub(r'\x96', '-', text)        # replace bullet
    text = re.sub(r'\x99', '™', text)  # replace TM
#     text = re.sub(r'\xae', '®', text)    # replace (R)
    text = re.sub(r'\xb0', '°', text)    # replace degree symbol
    text = re.sub(r'\xba', '°', text)    # replace degree symbol
    text = re.sub(r'\x97', ' ', text)
    text = re.sub(r'\x92', ' ', text)

    # Do you want to keep new lines / carriage returns? These are generally 
    # okay and useful for readability
    #text = re.sub(r'[\n\r]+', ' ', text)     # remove embedded \n and \r

    # This is a hard-core line that strips everything else.
    text = re.sub(r'[\x00-\x1f\x80-\xff]', ' ', text)

    return text


# In[62]:


def data_reading(data_path):
    data_files = os.listdir(data_path)
    cleaned_text = []
#     read all of the .txt files in the folder
#   clean raw text using processing(cleanText("file.txt"))

    for t in data_files:
        if t.split('.')[-1] == 'txt':
            cleaned_text.append(processing(cleanText( open(data_path + '/' + t, 'r').read())))

    return cleaned_text


# In[91]:



def word_count(cleaned_text, word_freq = 25):
# build word count matrix, delete words that appear less than word_frequency
    
#   first build the total word count matrix
    vect = CountVectorizer(min_df=1, token_pattern ='\\b\\w+\\b')
    cvt_matrix = vect.fit_transform(cleaned_text).toarray()
    cvt_features = vect.get_feature_names()
#   remove words that appear less than word_freq, or too short
    total = np.sum(cvt_matrix,axis=0)
    vital_vocab=[]
    vital_vocab_idx=[]
    for i in range(len(total)):
        cur_freq = total[i]
        word = cvt_features[i]
        if cur_freq >= word_freq and word not in stop_words:
            if len(word) >2: 
                vital_vocab.append(word)
                vital_vocab_idx.append(i)
                
    vital_count_mat = np.zeros((len(cleaned_text),len(vital_vocab)))
    
    for j in range(len(vital_vocab_idx)):
        idx = vital_vocab_idx[j]
        vital_count_mat[:,j] = cvt_matrix[:,idx]
        
    final_mat = vital_count_mat
    rel_docs = np.sum(final_mat, axis = 1)
    ret_idx = []
    
    for i in range(len(rel_docs)):
        relv = rel_docs[i]
        if relv > 0:
            ret_idx.append(i)
    final_mat = final_mat[ret_idx, :]

    return cvt_matrix, cvt_features, vital_vocab, vital_vocab_idx, final_mat


# In[97]:



def LSA_matrix(count_mat, dim, tfmodel = None, svdmodel = None):
    if not tfmodel:
        tfmodel = TfidfTransformer()
        tf_matrix = tfmodel.fit_transform(count_mat)
    else:
        tf_matrix = tfmodel.transform(count_mat)
    if not svdmodel:
        svdmodel = TruncatedSVD(dim)
        lsa = svdmodel.fit_transform( np.transpose( tf_matrix ) )
    else:
        lsa = svdmodel.transform( np.transpose( tf_matrix ))
    
    return lsa, tfmodel, svdmodel
    


# In[99]:


def universal_vector(word_embedding, source, local_vocab):
    glv_vec = []
    ids = []
    for i in range(len(local_vocab)):
        try:
            tmp = word_embedding.get_vector(source = source, word = local_vocab[i])
            glv_vec.append(tmp)
            ids.append(i)
        except KeyError:
            continue
    both_vocab = np.take(local_vocab, ids)
    return np.array(glv_vec), both_vocab, ids


# In[102]:


def cca(vocab1, vocab2, cca_model = None, dim = 300, max_iter = 1000, thre = 0.5):
    if not cca_model:
        cca_model = CCA(n_components=dim,max_iter=max_iter)
        try:
            cca_model.fit(vocab1, vocab2)
            [cca_vec1,cca_vec2] = cca_model.transform(vocab1,vocab2)
        except :
            print ('svd cannot converge, try smaller dim')
    else:
        [cca_vec1,cca_vec2] = cca_model.transform(vocab1,vocab2)
    comb_cca = (thre * cca_vec1+ (1-thre) * cca_vec2)
    return comb_cca, cca_vec1, cca_vec2, cca_model
    


# In[103]:


def rcca(vocab1, vocab2, rcca_model = None, dim = 300, guess_sig = 1, reg = 0.01, thre = 0.5):
    if not rcca_model:
        rcca_model = rccaMod.CCA(reg=reg,numCC=dim,kernelcca=True,ktype="gaussian",
                  gausigma=guess_sig)
        cancomps=rcca_model.train([vocab1,vocab2]).comps
        vec1 = cancomps[0]
        vec2 = cancomps[1]
    else:
        print ('sorry we are working on it')
    rcca_vec1 = [r1 / np.linalg.norm(r1) for r1 in vec1]
    rcca_vec2 = [r2 / np.linalg.norm(r2) for r2 in vec2]
    rcca_vec1 = np.array(rcca_vec1)
    rcca_vec2 = np.array(rcca_vec2)
    comb_rcca = (thre * rcca_vec1+ (1-thre) * rcca_vec2)
    return comb_rcca, rcca_vec1, rcca_vec2, rcca_model


# In[105]:


def most_change_word(cca_vector, universe_vector,both_vocab, N = 10):

    check_list = []
    for i in range(len(both_vocab)):
        check_list.append(distance.cosine(cca_vector[i], universe_vector[i]))
    sort_list = np.argsort(check_list)
    return np.take(both_vocab, sort_list[-N:])
    


# In[106]:


#def most_similar_word(cca_vector, universe_vector,both_vocab, word_id, N = 10):
#    word = both_vocab[word_id]
#
#    universal_list = []
#    cca_list = []
#
#    for i in range(len(both_vocab)):
#        if str(i) != word_id:
#            universal_list.append(distance.cosine(universe_vector[word_id], universe_vector[i]))
#            cca_list.append(distance.cosine(cca_vector[word_id], cca_vector[i]))
#    sort_list_1 = np.argsort(cca_list)
#    sort_list_2 = np.argsort(universal_list)
#
#    return np.take(both_vocab, sort_list_1[0: N]), np.take(both_vocab, sort_list_2[0:N])

def most_similar_word(cca_vector, universe_vector, both_vocab, word_id, N=10):
    word = both_vocab[word_id]
    
    universal_list = []
    cca_list = []
    
    for i in range(len(both_vocab)):
        
        universal_list.append(distance.cosine(universe_vector[word_id], universe_vector[i]))
        cca_list.append(distance.cosine(cca_vector[word_id], cca_vector[i]))
    sort_list_1 = np.argsort(cca_list)
    sort_list_2 = np.argsort(universal_list)
    return np.take(both_vocab, sort_list_1[0: N]), np.take(both_vocab, sort_list_2[0:N])

def vector_saving( cur_path, cca_vector, univ_cca, lsa_cca, task_id,mode = 'cca'):
    np.save(cur_path+'/outputs/vectors/'+str(mode) + '_output_' + str(task_id), cca_vector)
    np.save(cur_path+'/outputs/vectors/'+str(mode) + '_universal_output_' + str(task_id), univ_cca)
    np.save(cur_path+'/outputs/vectors/'+str(mode) + '_lsa_output_' + str(task_id), lsa_cca)



def model_saving(cur_path, svd_model, tfidf_model, model, task_id, mode):
    dump(svd_model, cur_path + '/outputs/models/svdmodel_' + str(task_id) +'.joblib') 
    dump(tfidf_model, cur_path + '/outputs/models/tfidfmodel_' + str(task_id) +'.joblib')
    if mode == 'cca':
        dump(model, cur_path + '/outputs/models/ccamodel_' + str(task_id) +'.joblib')
    elif mode == 'rcca':    
        dump(model, cur_path + '/outputs/models/rccamodel_' + str(task_id) +'.joblib')

def words_saving(cur_path, total_words, vital_words, both_words, task_id):
    f1 = open(cur_path+'/outputs/words/total_words_' + str(task_id) + '.txt', 'w' )
    for i in range(len(total_words)):
        f1.write(str(i) + '. ')
        f1.write(total_words[i] + '\n')
    f1.close()
    
    f2 = open(cur_path+'/outputs/words/vital_words_' + str(task_id) + '.txt', 'w' )
    for j in range(len(vital_words)):
        f2.write(str(j) + '. ')
        f2.write(vital_words[j] + '\n')
    f2.close()
    
    f3 = open(cur_path+'/outputs/words/both_words_' + str(task_id) + '.txt', 'w' )
    for k in range(len(both_words)):
        f3.write(str(k) + '. ')
        f3.write(both_words[k] + '\n')
    f3.close()


def words_analysis_saving(cur_path, most_change_words, most_similar_words_cca, 
                          most_similar_words_uni, word, task_id):
    f1 = open(cur_path+'/outputs/words_analysis/most_change_' + str(task_id) + '.txt', 'w' )
    for w in range(len(most_change_words)):
        f1.write(str(w) + '. ')
        f1.write(most_change_words[w] + '\n')
    f1.close()
    
    f2 = open(cur_path+'/outputs/words_analysis/similar_cca_' + str(task_id) + '_' + str(word) + '.txt', 'w' )
    for w in most_similar_words_cca:
        f2.write(w + '\n')
    f2.close()
    
    f3 = open(cur_path+'/outputs/words_analysis/similar_universal_' + str(task_id) + '_' +  str(word) + '.txt', 'w' )
    for w in most_similar_words_uni:
        f3.write(w + '\n')
    f3.close()
