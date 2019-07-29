#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os
import numpy as np
from wordembedding_cls import WordEmbedding
from tools import data_reading, word_count, LSA_matrix
from tools import universal_vector, cca, rcca, most_change_word,most_similar_word
from tools import vector_saving, model_saving, words_saving, words_analysis_saving


# In[18]:


from constant import task_id, word_model, source, train_one, train_two, mode, dim, convert
from constant import max_iter, threshold, guess_sig, reg, analysis_one, analysis_two,word_id, N


# In[2]:


def main():
    cur_path = os.getcwd()
    if train_one:
        print ('Begin Training the cca model')
        print ('---------------')
        print ('Start loading word embedding model')
        word_embedding = WordEmbedding()
        if convert :
            word_embedding.convert(source = source, input_file_path = cur_path + '/glove.42B.300d.txt',output_file_path = cur_path + '/' + str(word_model))
        word_embedding.load(source=source, file_path=cur_path + '/' + str(word_model))
        print ('Finish loading word embedding model')
        print ('---------------')
        print ('Start processing data and building LSA')
        cleaned_text = data_reading(cur_path + '/data/train_one')
        cvt_matrix, cvt_features, vital_vocab, vital_vocab_idx, final_mat = word_count(cleaned_text)
        lsa,tfmodel,svdmodel = LSA_matrix(final_mat, dim = dim)
        print ('Finish processing data and building LSA')
        print ('---------------')

        glv_vec,both_vocab,ids = universal_vector(word_embedding, source, vital_vocab)
        lsa_vec = lsa[ids, :]
        print ('Start training cca or rcca model')
        if mode == 'cca':
            comb_vec, vec1, vec2, model = cca(glv_vec, lsa_vec, dim = dim, max_iter = max_iter, thre = threshold)
        elif mode == 'rcca':
            comb_vec, vec1, vec2, model = rcca(glv_vec, lsa_vec,dim = dim, guess_sig = guess_sig, reg = reg, thre = threshold)
        else:
            print ('Please choose cca or rcca')
        print ('Finish processing cca or rcca model')
        print ('---------------')
        print ('Start saving model and cca rcca vectors')
        if mode == 'cca':
            vector_saving(cur_path, comb_vec, vec1, vec2, task_id, mode)
        elif mode == 'rcca':
            vector_saving(cur_path, comb_vec, vec1, vec2, task_id, mode)
        model_saving(cur_path, svdmodel, tfmodel, model, task_id, mode)
        print ('Finish saving model and cca rcca vectors')
        print ('---------------')
        print ('Start saving word vocabs')
        words_saving(cur_path, cvt_features, vital_vocab, both_vocab, task_id)
        print ('Finish saving word vocabs')
        print ('---------------')
    if train_two:
        print ('Begin Training the cca model')
        print ('---------------')
        print ('Start loading word embedding model')
        word_embedding = WordEmbedding()
        if convert :
            word_embedding.convert(source = source, input_file_path = cur_path + '/glove.42B.300d.txt',output_file_path = cur_path + '/' + str(word_model))
        word_embedding.load(source=source, file_path=cur_path + '/' + str(word_model))
        print ('Finish loading word embedding model')
        print ('---------------')
        print ('Start processing data and building LSA')
        cleaned_text_1 = data_reading(cur_path + '/data/train_two/first')
        cleaned_text_2 = data_reading(cur_path + '/data/train_two/second')
        idx = '0'
        for cleaned_text in [cleaned_text_1, cleaned_text_2]:
            idx = int(idx) + 1
            idx = str(idx)
            cvt_matrix, cvt_features, vital_vocab, vital_vocab_idx, final_mat = word_count(cleaned_text)
            lsa,tfmodel,svdmodel = LSA_matrix(final_mat, dim = dim)
            print ('Start processing the ' + str(idx) + ' domain' )
            print ('Finish processing data and building LSA')
            print ('---------------')

            glv_vec,both_vocab,ids = universal_vector(word_embedding, source, vital_vocab)
            lsa_vec = lsa[ids, :]
            print ('Start training cca or rcca model')
            if mode == 'cca':
                comb_vec, vec1, vec2, model = cca(glv_vec, lsa_vec, dim = dim, max_iter = max_iter, thre = threshold)
            elif mode == 'rcca':
                comb_vec, vec1, vec2, model = rcca(glv_vec, lsa_vec,dim = dim, guess_sig = guess_sig, reg = reg, thre = threshold)
            else:
                print ('Please choose cca or rcca')
            print ('Finish processing cca or rcca model')
            print ('---------------')
            print ('Start saving model and cca rcca vectors')
            if mode == 'cca':
                vector_saving(cur_path, comb_vec, vec1, vec2, task_id+'_'+idx, mode)
            elif mode == 'rcca':
                vector_saving(cur_path, comb_vec, vec1, vec2, task_id+'_'+idx, mode)
            model_saving(cur_path, svdmodel, tfmodel, model, task_id+'_'+idx, mode)
            print ('Finish saving model and cca rcca vectors')
            print ('---------------')
            print ('Start saving word vocabs')
            words_saving(cur_path, cvt_features, vital_vocab, both_vocab, task_id+'_'+idx)
            print ('Finish saving word vocabs')
            print ('---------------')
          
        print ('Start processing domain 1 and domain 2')
        comb_vec_1 = np.load(cur_path + '/outputs/vectors/'+str(mode) + '_output_' + str(task_id)+'_1' + '.npy')
        comb_vec_2 = np.load(cur_path + '/outputs/vectors/'+str(mode) + '_output_' + str(task_id)+'_2' + '.npy')
        vocab1 = open(cur_path + '/outputs/words/both_words_' + str(task_id) + '_1' + '.txt', 'r').read().split('\n')[0:-1]
        vocab2 = open(cur_path + '/outputs/words/both_words_' + str(task_id) + '_2' + '.txt', 'r').read().split('\n')[0:-1]
        vocab_1 = []
        vocab_2 = []
        for v in vocab1:
            vocab_1.append(v.split(' ')[-1])
        for v in vocab2:
            vocab_2.append(v.split(' ')[-1])       
        new_vec_1 = []
        new_vec_2 = []
        shared_vec = []
        print('load successfully')
        
        for i in range(len(vocab_1)):
            if vocab_1[i] in vocab_2:   
                shared_vec.append(vocab_1[i])
                new_vec_1.append(comb_vec_1[i])
                idx_2 = vocab_2.index(vocab_1[i])
                new_vec_2.append(comb_vec_2[idx_2])
        new_vec_1 = np.array(new_vec_1)
        new_vec_2 = np.array(new_vec_2)
        print (new_vec_1.shape)
        if mode == 'cca':
            comb_vec, vec1, vec2, model = cca(new_vec_1, new_vec_2, dim = dim, max_iter = max_iter, thre = threshold)
        elif mode == 'rcca':
            comb_vec, vec1, vec2, model = rcca(new_vec_1, new_vec_2,dim = dim, guess_sig = guess_sig, reg = reg, thre = threshold)
        else:
            print ('Please choose cca or rcca')
        f = open(cur_path + '/outputs/words/both_words_' + str(task_id) + '_12' + '.txt', 'w')
        np.save(cur_path+'/outputs/vectors/'+str(mode) + '_traintwo_' + str(task_id), comb_vec)
        np.save(cur_path+'/outputs/vectors/'+str(mode) + '_traintwo_one_' + str(task_id), vec1)
        np.save(cur_path+'/outputs/vectors/'+str(mode) + '_traintwo_two_' + str(task_id), vec2)
        np.save(cur_path+'/outputs/vectors/'+str(mode) + '_traintwo_one_oo' + str(task_id), new_vec_1)
        np.save(cur_path+'/outputs/vectors/'+str(mode) + '_traintwo_two_oo' + str(task_id), new_vec_2)
        f1 = open(cur_path+'/outputs/words/total_words_traintwo_' + str(task_id) + '.txt', 'w' )
        for i in range(len(shared_vec)):
            f1.write(str(i) + '. ')
            f1.write(shared_vec[i] + '\n')
        f1.close()
        


        
    if analysis_one:
        comb_vec = np.load(cur_path + '/outputs/vectors/'+str(mode) + '_output_' + str(task_id) + '.npy')
        glv_vec = np.load(cur_path + '/outputs/vectors/'+str(mode) + '_universal_output_' + str(task_id) + '.npy')
        vocab = open(cur_path + '/outputs/words/both_words_' + str(task_id) + '.txt', 'r').read()
        both_vocab = vocab.split('\n')[0:-1]
        print ('Start searching most change words')
        change_result = most_change_word(comb_vec, glv_vec, both_vocab, N)
        print ('Finish searching most change words')
        print ('---------------')
        print ('Start finding most common words')
        similar_result_1, similar_result_2  = most_similar_word(comb_vec, glv_vec, both_vocab, word_id, N = N)

        print ('Finish finding most common words')
        print ('---------------')
        print ('Start saving word analysis result')
        words_analysis_saving(cur_path, change_result, similar_result_1, similar_result_2, both_vocab[word_id], task_id )
        print ('Finish saving word analysis result')
        print ('---------------')
    if analysis_two:
        vec_1 = np.load(cur_path + '/outputs/vectors/' + str(mode) + '_traintwo_one_oo' + str(task_id) + '.npy')
        vec_2 = np.load(cur_path + '/outputs/vectors/' + str(mode) + '_traintwo_two_oo' + str(task_id) + '.npy')
        words = open(cur_path+'/outputs/words/total_words_traintwo_' + str(task_id) + '.txt', 'r' ).read().split('\n')[0:-1]

        change_result = most_change_word(vec_1, vec_2, words, N)
        similar_result_1, similar_result_2  = most_similar_word(vec_1, vec_2, words, word_id, N = N)
        f1 = open(cur_path+'/outputs/words_analysis/most_change_twodomain' + str(task_id) + '.txt', 'w' )
        for w in range(len(change_result)):
            f1.write(str(w) + '. ')
            f1.write(change_result[w] + '\n')
        f1.close()
        
        f2 = open(cur_path+'/outputs/words_analysis/similar_traintwo_firstdomain' + str(task_id) + '_' + str(words[word_id]) + '.txt', 'w' )
        for w in similar_result_1:
            f2.write(w + '\n')
        f2.close()
        
        f3 = open(cur_path+'/outputs/words_analysis/similar_traintwo_seconddomain_' + str(task_id) + '_' +  str(words[word_id]) + '.txt', 'w' )
        for w in similar_result_2:
            f3.write(w + '\n')
        f3.close()

# In[ ]:


if __name__ == '__main__':
    main()


# cur_path = os.getcwd()
# if train:
#     print ('Begin Training the cca model')
#     print ('---------------')
#     print ('Start loading word embedding model')
#     word_embedding = WordEmbedding()
#     word_embedding.load(source=source, file_path=cur_path + '/' + str(word_model))
#     print ('Finish loading word embedding model')
#     print ('---------------')
#     print ('Start processing data and building LSA')
#     cleaned_text = data_reading(cur_path)
#     cvt_matrix, cvt_features, vital_vocab, vital_vocab_idx, final_mat = word_count(cleaned_text)
#     lsa,tfmodel,svdmodel = LSA_matrix(final_mat, dim = dim)
#     print ('Finish processing data and building LSA')
#     print ('---------------')
    
#     glv_vec,both_vocab,ids = universal_vector(word_embedding, source, vital_vocab)
#     lsa_vec = lsa[ids, :]
#     print ('Start training cca or rcca model')
#     if mode == 'cca':
#         comb_vec, vec1, vec2, model = cca(glv_vec, lsa_vec, dim = dim, max_iter = max_iter, thre = threshold)
#     elif mode == 'rcca':
#         comb_vec, vec1, vec2, model = rcca(glv_vec, lsa_vec,dim = dim, guess_sig = guess_sig, reg = reg, thre = threshold)
#     else:
#         print ('Please choose cca or rcca')
#     print ('Finish processing cca or rcca model')
#     print ('---------------')
#     print ('Start saving model and cca rcca vectors')
#     if mode == 'cca':
#         vector_saving(cur_path, comb_vec, vec1, vec2, task_id, mode)
#     elif mode == 'rcca':
#         vector_saving(cur_path, comb_vec, vec1, vec2, task_id, mode)
#     model_saving(cur_path, svdmodel, tfmodel, model, task_id, mode)
#     print ('Finish saving model and cca rcca vectors')
#     print ('---------------')
#     print ('Start saving word vocabs')
#     words_saving(cur_path, cvt_features, vital_vocab, both_vocab, task_id)
#     print ('Finish saving word vocabs')
#     print ('---------------')


# In[15]:


# if analysis:
#     print ('Start searching most change words')
#     change_result = most_change_word(comb_vec, glv_vec, both_vocab, N)
#     print ('Finish searching most change words')
#     print ('---------------')
#     print ('Start finding most common words')
#     similar_result_1, similar_result_2  = most_similar_word(comb_vec, glv_vec, both_vocab, word_id)
#     print ('Finish finding most common words')
#     print ('---------------')
#     print ('Start saving word analysis result')
#     words_analysis_saving(cur_path, change_result, similar_result_1, similar_result_2, both_vocab[word_id], task_id )
#     print ('Finish saving word analysis result')
#     print ('---------------')
    
    


# In[ ]:




