import pandas as pd
import numpy as np


import tensorflow as tf

from data_prepare import *

def load_data(train_file, test_file):
    col_name = ['index', 's1', 's2', 'label']
    train_data = pd.read_csv(train_file, sep='\t', header=None, names=col_name)
    test_data = pd.read_csv(train_file, sep='\t', header=None, names=col_name)

    return train_data, test_data

def save_data(feature_train, feature_test, col_names, feature_name):
    project.save_features(feature_train, feature_test, col_names, feature_name)


def extract_feature_siames_lstm_ManDist():
    pass

def extract_feature_siames_lstm_attention():
    pass

def extract_feature_siames_lstm_dssm():
    pass

def extract_sentece_length_diff():
    """
    长度差特征
    :return:
    """
    pass

def extract_edit_distance():
    """
    抽取编辑距离
    :return:
    """
    pass

def extract_ngram():
    """
    提取ngram特征
    :return:
    """
    pass

def extract_sentence_diff_same():
    """
    抽取两个句子的相同和不同的词特征
    :return:
    """
    pass

def extract_doubt_sim():
    """
    抽取疑问词相同的比例
    :return:
    """
    pass


def extract_sentence_exist_topic():
    """
    抽取两个句子是否包含相同的主题
    :return:
    """

    pass

def extract_word_embedding_sim(w2v_model_path='train_all_data.bigram')
    """
    提取句子的词向量组合相似度
    :param w2v_model_path: 
    :return: 
    """
    pass

if __name__ == '__main__':
    pass

