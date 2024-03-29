import numpy as np
from numpy import array
import tensorflow as tf
import keras
import scipy
import matplotlib.pyplot as plt
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras import optimizers
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras import metrics
import config
import os
import logging
import multiprocessing
from joblib import Parallel, delayed
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.layers.normalization import BatchNormalization
from pre_processing_ref import load_data_from_csv, seg_words, train_vec, train_tencent_model, convert_to_onehot, sentence_to_indice, embedding_data, wordToIndex, convert_prob_to_cls_name, convert_oh_to_cls_name, get_uniq_index, new_indice

# define documents
train=load_data_from_csv(config.train_data_path)
val = load_data_from_csv(config.validate_data_path)
print("finish loading data")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)
#test = load_data_from_csv(config.test_data_path)

#val dataset as hold-out val set and test dataset; do cv on train dataset
m_val = val.shape[0]
train_doc = train.iloc[:, 1]
val_doc = val.iloc[0:7500, 1]
test_doc = val.iloc[7501:m_val-1, 1]
'''train_doc = train.iloc[1:10, 1]
val_doc = val.iloc[0:5, 1]
test_doc = val.iloc[6:10, 1]'''
logger.info("finish splitting")
logger.info(train_doc.shape)
logger.info(val_doc.shape)
logger.info(test_doc.shape)

# define class labels
#train_labels = array(train[:, 2:])
#val_labels = array(val[0:7500, 2:])
#test_labels = array(val[7501:m_val-1, 2:])
train_labels = array(train.iloc[:, 2:])
val_labels = array(val.iloc[0:7500, 2:])
test_labels = array(val.iloc[7501:m_val-1, 2:])

#sub_select columns for specific topics
'''column_index = np.array([0, 2, 3, 6, 7, 15, 17, 19])
column_title = dict()
column_title[0]="location_traffic_convenience"
column_title[1]="location_easy_to_find"
column_title[2]="service_wait_time"
column_title[3]="service_serving_speed"
column_title[4]="price_level"
column_title[5]="dish_taste"
column_title[6]="dish_recommendation"
column_title[7]="others_willing_to_consume_again"

train_labels = train_labels[:, column_index]
val_labels = val_labels[:, column_index]
test_labels = test_labels[:, column_index]

#print("test number"+str(test_labels.shape[0]))
logger.info("finish loading sliced labels")
logger.info(train_labels.shape)
logger.info(val_labels.shape)
logger.info(test_labels.shape)'''

train_indices = np.load("train_indices_tenc.dat")
val_indices = np.load("val_indices_tenc.dat")
test_indices = np.load("test_indices_tenc.dat")

#debug part
'''train_indices = train_indices[1:20, :]
val_indices = val_indices[1:10, :]
test_indices = test_indices[11:20, :]

train_labels=train_labels[1:20,:]
val_labels=val_labels[1:10, :]
test_labels=test_labels[11:20,:]'''

max_length=train_indices.shape[1]
logger.info("max_length")
logger.info(max_length)

index_appear = get_uniq_index(train_indices, val_indices, test_indices)
ind_list = index_appear.tolist()

#change to small_emb indices
train_indices = new_indice(train_indices, ind_list)
val_indices = new_indice(val_indices, ind_list)
test_indices = new_indice(test_indices, ind_list)

index_new_set = get_uniq_index(train_indices, val_indices, test_indices)
ind_new_list = index_new_set.tolist()
logger.info("new_set length: " + str(len(ind_new_list)))
jjjjjj=ind_new_list[len(ind_new_list)-1]
logger.info("max value: "+str(jjjjjj))

embedding_matrix=np.load("/scratch/users/qingyin/output/small_embeddinge_matrix_tenc.dat")
vocab_len = embedding_matrix.shape[0]
emb_dim = embedding_matrix.shape[1]
logger.info("vocab_len")
logger.info(vocab_len)
logger.info("emb_dim")
logger.info(emb_dim)
#print(embedding_matrix[2, :])

# define the model
DROPOUT=0.5
NUM_HID=128
REC_DROP=0.5
n = train_labels.shape[1]

#y_train=np.array(convert_to_onehot(train_labels[100:150, 1], 4))

#print(y_train)
#model=dict()
def loopone(i):
	def f1(y_true, y_pred):
		y_pred_label = convert_prob_to_cls_name(y_pred)
		y_true_label = convert_oh_to_cls_name(y_true)
		#print(y_pred_label)
		#print(y_true_label)
		f1score = f1_score(y_true_label, y_pred_label, average='weighted')
		f1_ave = np.average(f1score)
		return f1_ave
	logger.info("start train column" +str(i))
	if i == 5 or i == 18:
		class_weight = {
		0: 14.0/37.0,
		1: 14.0/37.0,
		2: 5.0/37.0,
		3: 4.0/37.0
		}
	else:
		class_weight = {
		0: 1.0/37.0,
		1: 12.0/37.0,
		2: 12.0/37.0,
		3: 12.0/37.0
		}
	model = Sequential()
	#model.add(Input(shape=(max_length,), dtype='int32'))
	model.add(Embedding(vocab_len, emb_dim, input_length=max_length, weights=[embedding_matrix], trainable=False))
	model.add(LSTM(NUM_HID, dropout=DROPOUT, recurrent_dropout=REC_DROP,return_sequences=True))
	model.add(LSTM(NUM_HID, dropout=DROPOUT, recurrent_dropout=REC_DROP,return_sequences=False))
	#model.add(Flatten())
	model.add(Dense(4))
	model.add(BatchNormalization())
	model.add(Activation('softmax'))
	# compile the model
	adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
	# summarize the model
	print(model.summary())
	# fit the model
	n = train_labels.shape[1]
	#one_hot y
	y_train=np.array(convert_to_onehot(train_labels[:, i], 4))
	y_val=np.array(convert_to_onehot(val_labels[:, i], 4))
	#print(y)
	#val_f1 = f1(y_val, model.predict(val_indices, verbose = 2))
	#early_stopping =EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=False)
	history = model.fit(train_indices, y_train, validation_data=(val_indices, y_val),class_weight=class_weight, batch_size = 128, epochs=20, verbose=2) #, callbacks = [early_stopping])
	# Plot training & validation accuracy values
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title(str(i)+'th Tenc Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(str(i)+'_acc_tenc.png')

	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title(str(i)+'th Tenc Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(str(i)+'_loss_tenc.png')
	# evaluate the model
	y_test=np.array(convert_to_onehot(test_labels[:, i], 4))
	y_test_pred = model.predict(test_indices, verbose=2)
	#print(str(i)+"th y_test_pred shape: ")
	#print(y_test_pred.shape)
	#print(y_test_pred)
	f1_test = f1(y_test, y_test_pred)
	logger.info(str(i)+"th f1 score: ")
	print(str(i)+"th f1 score: ")
	print(f1_test)
	#logger.info(f1_test)

	#logger.info("model "+str(i)+ model.metrics_names)
	scores = model.evaluate(test_indices, y_test, verbose=2)
	logger.info("model "+str(i) + ":")
	print("model "+str(i) + "metrics:")
	print(model.metrics_names)
	print(scores)
	#print(str(i)+'th model w2v '+'Accuracy: %f' % (accuracy*100))
	#logger.info(str(i)+'th model tenc '+'Accuracy: %f' % (accuracy*100))
	#logger.info(str(i)+'th model tenc f1_score:' + f1_score(y_test, y_test_pred))
	logger.info("complete train model" + str(i))
	model.save("/scratch/users/qingyin/output/" +str(i) + 'th tenc model.h5')
	print("complete saving model " + str(i))
	del model
	#model[i] = model #store model to dict
num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(loopone)(i) for i in range(n))
logger.info("complete train model")
