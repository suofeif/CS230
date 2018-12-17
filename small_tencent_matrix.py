import numpy as np
from numpy import array
import tensorflow as tf
import keras
import config
import os
import logging
import multiprocessing
import pickle
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from pre_processing_ref import load_data_from_csv, seg_words, train_vec, train_tencent_model, convert_to_onehot, sentence_to_indice, embedding_data, wordToIndex, get_uniq_index

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

train_indices = np.load("train_indices_tenc.dat")
val_indices = np.load("val_indices_tenc.dat")
test_indices = np.load("test_indices_tenc.dat")
max_length=train_indices.shape[1]
logger.info("max_length")
logger.info(max_length)

embedding_matrix=np.load("/scratch/users/qingyin/output/embedding_matrix_tenc.dat")
vocab_len = embedding_matrix.shape[0]
emb_dim = embedding_matrix.shape[1]
logger.info("vocab_len")

index_appear = get_uniq_index(train_indices, val_indices, test_indices)
logger.info("unique length: " + str(index_appear.size))
small_emb = embedding_matrix[index_appear, :]
logger.info("small matrix size: " + str(small_emb.shape[0]))

with open('/scratch/users/qingyin/output/small_embeddinge_matrix_tenc.dat', 'wb') as outfile:
    pickle.dump(small_emb, outfile, pickle.HIGHEST_PROTOCOL)

logger.info("finish saving small_emb")
