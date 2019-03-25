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
    # step1 定义抽取特征的方式名
    feature_name = 'nlp_edit_distance'

    # step2 载入数据
    train_data, test_data = load_data(
        train_file=project.data_dir + 'atec_nlp_sim_train_0.6.csv',
        test_file=project.data_dir + 'atec_nlp_sim_test_0.4.csv')

    feature_train = np.zeros([train_data.shape[0],1], dtype='float64')
    feature_test = np.zeros([test_data.shape[0],1], dtype='float64')

    def get_edit_distance(s1, s2):
        n,m = len(s1)+1, len(s2)+1
        matrix = np.zeros([n,m])
        matrix[0][0] = 0
        for i in range( 1,m):
            matrix[0][i] = matrix[0][i-1]+1
        for i in range(1, n):
            matrix[i][0] = matrix[i-1][0] +1
        # for i in range(matrix.shape[0]):
        #     print(matrix[i])
        for i in range(1,n):
            for j in range(1,m):
                if s1[i-1] == s2[j-1]:
                    cost = 0
                else:
                    cost = 1
                matrix[i][j] = min(matrix[i-1][j] +1, matrix[i][j-1] +1, matrix[i-1][j-1] + cost)
        # print('---------------------------')
        # for i in range(matrix.shape[0]):
        #     print(matrix[i])
        # print(matrix[n-1][m-1])
        return 1 - matrix[n-1][m-1] / max(len(s1), len(s2))
    # s1 = '明天会下雨'
    # s2 = '明天小明将会去打球'
    # print(get_edit_distance(s1, s2))

    for index, df in train_data.iterrows():
        s1 = df['s1'].strip()
        s2 = df['s2'].strip()
        edit_distance = get_edit_distance(s1, s2)
        feature_train[index] = round(edit_distance, 5)

    for index, df in test_data.iterrows():
        s1 = df['s1'].strip()
        s2 = df['s2'].strip()
        edit_distance = get_edit_distance(s1, s2)
        feature_train[index] = round(edit_distance, 5)

    col_names = [feature_name]
    save_data(feature_train, feature_test,col_names, feature_test)

    pass

def extract_ngram(max_ngram = 3):
    """
    提取ngram特征
    :return:
    """
    # step1 定义抽取特征的方式名
    feature_name = 'nlp_edit_distance'

    # step2 载入数据
    train_data, test_data = load_data(
        train_file=project.data_dir + 'atec_nlp_sim_train_0.6.csv',
        test_file=project.data_dir + 'atec_nlp_sim_test_0.4.csv')

    feature_train = np.zeros([train_data.shape[0], 1], dtype='float64')
    feature_test = np.zeros([test_data.shape[0], 1], dtype='float64')


    def get_ngram(s, ngram):
        result = []
        for i in range(len(s)):
            if i + ngram < len(s)+1:
                result.append(s[i:i+ngram])
        return result

    def get_ngram_sim(s1_ngram, s2_ngram):
        s1_dict = {}
        s2_dict = {}
        for word in s1_ngram:
            if word not in s1_dict:
                s1_dict[word] = 1
            else:
                s1_dict[word] = s1_dict + 1
        s1_count = np.sum([value for key, value in s1_dict.items()])

        for word in s2_ngram:
            if word not in s2_ngram:
                s2_dict[word]  = 1
            else:
                s2_dict[word] = s2_dict[word] + 1
        s2_count = np.sum([value for key, value in s2_dict.items()])

        s1_count_only = np.sum(value for key, value in s1_dict.items() if key not in s2_dict)

        s2_count_only = np.sum(value for key, value in s2_dict.items() if key not in s1_dict)

        # s1 s2 都有，计算差值
        s1_s2_count = np.sumO(abs(value - s2_dict[key]) for key, value in s1_dict.items() if key in s2_dict)

        all_count = s1_count + s2_count

        return 1 - (s1_count_only + s2_count_only + s1_s2_count + all_count) / float(all_count + 0.000001)

    for index, df in train_data:
        s1 = df['s1']
        s2 = df['s2']
        s1_ngram = get_ngram(s1, max_ngram)
        s2_ngram = get_ngram(s2, max_ngram)

        ngram_sim = get_ngram_sim(s1_ngram, s2_ngram)
        feature_train[index] = ngram_sim

    for index, df in test_data:
        s1 = df['s1']
        s2 = df['s2']
        s1_ngram = get_ngram(s1, max_ngram)
        s2_ngram = get_ngram(s2, max_ngram)

        ngram_sim = get_ngram_sim(s1_ngram, s2_ngram)
        feature_test[index] = ngram_sim

    # step 3 保存特征：参数有：训练集的特征，测试集的特征，抽取特征的方法的多列特征的列名，抽取特征的方式名
    col_names = [('{}_{}'.format(feature_name, ngram_value)) for ngram_value in range(max_ngram)]
    save_data(feature_train, feature_test, col_names, feature_name)

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

def extract_word_embedding_sim(w2v_model_path='train_all_data.bigram'):
    """
    提取句子的词向量组合相似度
    :param w2v_model_path: 
    :return: 
    """

    pass


    feature_name = 'nlp_word_embedding_sim'


if __name__ == '__main__':
    extract_edit_distance()
    pass

