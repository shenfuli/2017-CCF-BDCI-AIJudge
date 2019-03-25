# -*- encoding:utf-8 -*-
from collections import defaultdict

import numpy as np
import pandas as pd
from config.db_config import Config
from gensim.models import Word2Vec
from utils import LOGGER

config = Config()
def get_raw_documents():
    '''
    加载文章列表数据
    :return:  列表数据的list格式数据 ["xxx xxx xxx","abc dd bb"]
    '''
    df_all = pd.read_csv(config.data_csv_path, encoding='utf8')
    documents = df_all['content'].values
    LOGGER.log('documents number %d' % len(documents))
    return documents

def get_words_list(documents):
    '''
    获取word2vec 训练样本数据
    :param documents:  文章的list 格式:["word1 word2 word3", "word1 word2 word3"]
    :return: 格式： [["cat", "say", "meow"], ["dog", "say", "woof"]]
    '''
    texts = [[word for word in document.split(' ')] for document in documents]
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] >= 5] for text in texts]
    return texts

def get_len(words_list):
    '''
    返回所有行数
    :param words_list:
    :return:
    '''
    rows=len(words_list)
    return rows


def build_w2v(texts):
    '''
    构建 文本的矩阵向量
    :param texts:
    :return:
    '''
    LOGGER.log('Start get w2v feat..')
    rows = get_len(texts)
    w2v_feat = np.zeros((rows, config.w2v_dim))
    w2v_feat_avg = np.zeros((rows, config.w2v_dim))
    i = 0
    for line in texts:
        num = 0
        for word in line:
            num += 1
            vec = model[word]
            w2v_feat[i, :] += vec
        w2v_feat_avg[i, :] = w2v_feat[i, :] / num  # 一个句子的向量表示＝所有词和的平均数值
        i += 1
        if i % 1000 == 0:
            LOGGER.log(i)

    feat_w2v_df = pd.DataFrame(w2v_feat)
    feat_w2v_avg_df = pd.DataFrame(w2v_feat_avg)
    LOGGER.log('Save w2v and w2v_avg feat done!')
    return feat_w2v_df, feat_w2v_avg_df



if __name__=="__main__":
    # 生成word2vec 格式的矩阵
    words_list = get_words_list(get_raw_documents())
    rows = get_len(words_list)
    # 加载w2v 模型
    model = Word2Vec.load(config.model_w2v)
    # 针对样本数据构建w2v的特征
    feat_w2v_df,feat_w2v_avg_df = build_w2v(words_list)
    feat_w2v_df.to_csv(config.feat_w2v, index=None)
    feat_w2v_avg_df.to_csv(config.feat_w2v_avg, index=None)

############################ w2v ############################
