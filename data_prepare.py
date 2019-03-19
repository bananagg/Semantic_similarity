import pandas as pd
import numpy as np
import jieba

from collections import defaultdict
from gensim.models import word2vec
from gensim.models import KeyedVectors
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

import json

import os
import re

from util import Project


train_filename = 'atec_nlp_sim_train.csv'
train_all_filename = 'atec_nlp_sim_train_add.csv'
stopwords_filename='stop_words.txt'
spelling_corrections_filename = 'spelling_corrections.json'

#param
embedding_size = 300
max_sentence_length = 20
max_word_length = 25
# os.path.join(project.aux_dir,'fasttext','')
max_vovab_size = 100000

project = Project.init(os.getcwd(), create_dir=False)

def load_stopwordslist(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        stop_words = [line.strip() for line in f]
    return stop_words

def load_spelling_corrections(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        spelling_corrections = json.load(file)
    return file


def transform_other_word(str_txt, reg_dict):
    for token_str, replac_str in reg_dict.items():
        str_txt = str_txt.replace(token_str, replac_str)
    return str_txt


def process_save_embedding_wv(nfile, type = 1, isStore_ids = False):

    w2v_path = project.aux_dir + 'sgns.zhihu.bigram'
    if type == 2:
        w2v_path = project.aux_dir + 'train_all_data.bigram'

    tokenizer = Tokenizer(
        num_words=max_vovab_size,
        split=' ',
        lower=False,
        char_level=False,
        filters=''
    )

    # 加载所有的词汇表训练集和测试集
    pre_deal_train_df = pd.read_csv(project.preprocessed_data_dir + 'train_0.6_seg.csv',
                                    names=["index", "s1", "s2", "label"],
                                    header=None, encoding='utf-8',
                                    sep='\t')
    pre_deal_test_df = pd.read_csv(project.preprocessed_data_dir + 'test_0.4_seg.csv',
                                   names=["index", "s1", "s2", "label"],
                                   header=None, encoding='utf-8',
                                   sep='\t',
                                   )
    texts = []
    texts_s1_test = pre_deal_test_df['s1'].tolist()
    texts_s2_test = pre_deal_test_df['s2'].tolist()

    texts_s1_train = pre_deal_train_df['s1'].tolist()
    texts_s2_train = pre_deal_train_df['s2'].tolist()

    texts.extend(texts_s1_test)
    texts.extend(texts_s2_test)
    texts.extend(texts_s1_train)
    texts.extend(texts_s2_train)

    # print pre_deal_train_df.isnull().any()
    # print pre_deal_test_df.isnull().any()
    # 生成token词典
    # tests = [u'中 国', u'矿业 大学', u'不错哦']
    tokenizer.fit_on_texts(texts)

    # 生成各个词对应的index列表
    s1_train_ids = tokenizer.texts_to_sequences(texts_s1_train)
    s2_train_ids = tokenizer.texts_to_sequences(texts_s2_train)

    s1_test_ids = tokenizer.texts_to_sequences(texts_s1_test)
    s2_test_ids = tokenizer.texts_to_sequences(texts_s2_test)

    num_words_dict = tokenizer.word_index

    # 训练集的词汇表的词向量矩阵,行数为最大值+1,形式为：index->vec
    embedding_matrix = 1 * np.random.randn(len(num_words_dict) + 1, embedding_size)
    embedding_matrix[0] = np.random.randn(embedding_size)

    w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=False)

    if type == 3:
        w2v_path2 = project.aux_dir+'train_all_data.bigram'
        w2v_model2 = KeyedVectors.load_word2vec_format(w2v_path2, binary=False)

    count = 0
    for word, index in num_words_dict.items():
        if word in w2v_model.vocab:
            embedding_matrix[index] = w2v_model.word_vec(word)
            count = count + 1
        else:
            pass


    pass

def pre_train_char_w2v(binary=False):


    data_local_df = pd.read_csv(project.data_dir + train_all_filename, sep='\t', names=['index', 's1', 's2', 'label'])
    data_local_all_df = pd.read_csv(project.data_dir + train_all_filename, sep='\t', names=['index', 's1', 's2', 'label'])



    stopwords = load_stopwordslist(project.aux_dir + stopwords_filename)
    spelling_corrections = load_spelling_corrections(project.aux_dir + spelling_corrections_filename)
    compile = re.compile(r"\*+")
    data_df_list = [data_local_df, data_local_all_df]
    texts = []
    char_vocabs = {}
    char_index = 1

    for data_df in data_df_list:
        # print(data_df)
        for index, row in data_df.iterrows():
            # print(index, row)
            if index != 0 and index % 5000 == 0:
                print("{:,} sentence word embedding.".format(index))
            for col_name in ['s1', 's2']:
                # 替换掉脱敏的数字
                # re_str = re_object.subn(u"十一", unicode(row[col_name], 'utf-8'))
                re_str = compile.subn(u'十一', row[col_name])
                # print(re_str)
                # spell_corr_str = re_str[0]
                spell_corr_str = transform_other_word(re_str[0], spelling_corrections)
                # 把句子拆分成单字
                spell_corr_str = list(spell_corr_str)
                # print(spell_corr_str)
                for char in spell_corr_str:
                    if char not in char_vocabs and char not in stopwords and char.strip() != '':
                        char_vocabs[char] = char_index
                        char_index += 1
                texts.extend(spell_corr_str)

    print(texts)
    model = word2vec.Word2Vec(sentences=texts, size=300, window=3, workers=2)
    model.wv.save_word2vec_format(fname=project.aux_dir + 'train_char_all__data.bigram', binary= binary, fvocab=None)
    project.save(project.preprocessed_data_dir, model)

def preprocessing(df, fname):
    stopwords = load_stopwordslist(project.aux_dir + stopwords_filename)
    spelling_corrections = load_spelling_corrections(project.aux_dir + spelling_corrections_filename)

    re_compile = re.compile('\*+')
    vocabs = defaultdict(int)  # 记录词汇表词频
    for data_df in df:
        for index, row in data_df.iterrows():
            if index != 0 and index % 2000 == 0:
                print("{:,}  {}-sentence embedding.".format(index, fname))
            for col_name in ['s1', 's2']:
                re_str = re_compile.subn(u'十一', row[col_name])
                spelling_corr_str = transform_other_word(re_str[0], spelling_corrections)
                seg_str = seg_sentence(spelling_corr_str, stopwords)
                for word in seg_str:
                    vocabs[word] = vocabs[word] + 1
                    data_df.at[index, col_name] = seg_str
    data_df.to_csv(project.preprocessed_data_dir + '{}.csv'.format(fname), sep='\t', header=None, index=None,
                   encoding='utf-8')
    project.save(project.preprocessed_data_dir + '{}.pickle'.format(fname), vocabs)
    del data_df


def seg_sentence(sentence, stopwords):
    sentence_seged = jieba.cut(sentence.strip())
    out_str = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != ' ':
                out_str += word
                out_str += ' '
    return out_str

def pre_train_w2v(binary=False):
    """
        利用已经训练集训练词向量
        :param nfile_corpus:已经分好词的文本路径，如"train_segment.corpus"
        :param binary:将词向量表是否存储为二进制文件
        :return:
        """
    # 加载所有的词汇表训练集和测试集
    pre_deal_train_df = pd.read_csv(project.preprocessed_data_dir + 'train_0.6_seg.csv',
                                    names=["index", "s1", "s2", "label"],
                                    header=None, encoding='utf-8',
                                    sep='\t')
    pre_deal_test_df = pd.read_csv(project.preprocessed_data_dir + 'test_0.4_seg.csv',
                                   names=["index", "s1", "s2", "label"],
                                   header=None, encoding='utf-8',
                                   sep='\t',
                                   )
    texts = []
    texts_s1_test = [line.strip().split(" ") for line in pre_deal_test_df['s1'].tolist()]
    texts_s2_test = [line.strip().split(" ") for line in pre_deal_test_df['s2'].tolist()]

    texts_s1_train = [line.strip().split(" ") for line in pre_deal_train_df['s1'].tolist()]
    texts_s2_train = [line.strip().split(" ") for line in pre_deal_train_df['s2'].tolist()]

    texts.extend(texts_s1_test)
    texts.extend(texts_s2_test)
    texts.extend(texts_s1_train)
    texts.extend(texts_s2_train)

    model = word2vec.Word2Vec(sentences=texts, size=300, window=2, min_count=3, workers=2)
    model.wv.save_word2vec_format(fname=project.aux_dir + "train_all_data.bigram", binary=binary, fvocab=None)
    pass


if __name__ == '__main__':
    # project = Project.init(os.getcwd(), create_dir=False)
    print(project.data_dir)
    data_local_df = pd.read_csv(project.data_dir + train_all_filename, sep='\t', names=['index', 's1', 's2', 'label'])
    # print(data_local_df.head(5))
    data_local_all_df = pd.read_csv(project.data_dir + train_all_filename, sep='\t', names=['index', 's1', 's2', 'label'])
    # print(data_local_all_df.head())

    pre_train_char_w2v()
    # print(load_stopwordslist(project.aux_dir+stopwords_filename))

    preprocessing(data_local_df, 'train_0.6_seg')
    # preprocessing(data_test_df, 'test_0.4_seg')
    # preprocessing(data_all_df, 'data_all_seg')

    pre_train_w2v()

    process_save_embedding_wv('train_all_w2v_embedding_matrix.pickle', type=2, isStore_ids=True)
    # process_save_embedding_wv('zhihu_w2v_embedding_matrix.pickle',type=2,isStore_ids=False)
    # process_save_embedding_wv('zhihu_w2v_embedding_matrix.pickle',type=3,isStore_ids=False)

    # step 4 char wordembedding
    process_save_char_embedding_wv(isStore_ids=True)