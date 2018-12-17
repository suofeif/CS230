import numpy as np
from numpy import array
import config
import os
import logging
from pre_processing_ref import load_data_from_csv, seg_words, train_vec, train_tencent_model, convert_to_onehot, sentence_to_indice, embedding_data, wordToIndex, convert_prob_to_cls_name, convert_oh_to_cls_name, get_uniq_index

train_indices = np.load("train_indices_tenc.dat")
val_indices = np.load("val_indices_tenc.dat")
test_indices = np.load("test_indices_tenc.dat")

index_appear = get_uniq_index(train_indices, val_indices, test_indices)

ind_list = index_appear.tolist()

train_indices = new_indice(train_indices, ind_list)
val_indices = new_indice(val_indices, ind_list)
test_indices = new_indice(test_indices, ind_list)

m = train_indices.shape[0]
n = train_indices.shape[1]
train = np.zeros((m, n))
for i in range(m):
    sentence = train_indices[i, :]
    for j in range(n):
        k = ind_list.index(sentence[j])
        train[i, j] = k

print(train)
