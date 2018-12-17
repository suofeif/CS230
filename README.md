
Fine-Grained Sentiment Analysis of Restaurant Customer Reviews in Chinese Language
=========================================

* The baseline model (main_train.py, main_predict.py, model.py, data_process.py, config.py) is provided by AI Challenger official. We trained the SVC with rbf kernel. Here is the link to official site: https://github.com/AIChallenger/AI_Challenger_2018/tree/master/Baselines/sentiment_analysis2018_baseline.
* All the other code files are produced by the team members.

Simple instructions:
---
* Datasets are available at: https://challenger.ai/competition/fsauor2018
* prc_save_Word2Vec.py load and output embedding matrix and indices files for train, val, and test data for LSTM-128A.
* pre_processing_ref.py provides utility functions for all the other files.
* keras_rnn.py is for LSTM-128A model using w2v representations.
* Continue_w2v.py is for further tuning and training on a subset of elements with lower performance in the output of keras_rnn.py (LSTM-256).
* prc_save_data.py for importing tencent w2v model and produce corresponding indices and embedding matrix.
* small_tencent_matrix.py outputs a subset of the embedding matrix from prc_save_data.py for the favor of computation.
* tenc_change_indices.py change all the indices from prc_save_data.py to be consistent with the embedding matrix from small_tencent_matrix.py.
* RNN_tencent_small.py is for LSTM-128B model using mini Tencent embedding model.
