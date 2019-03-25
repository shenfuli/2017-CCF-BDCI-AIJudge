# -*- encoding:utf-8 -*-
from collections import defaultdict
import pandas as pd
from config.db_config import Config
from utils import LOGGER
from gensim.models import Word2Vec
config = Config()

def get_raw_documents():
    '''
    加载文章列表数据
    :return:  列表数据的list格式数据 ["xxx xxx xxx","abc dd bb"]
    '''
    df_all = pd.read_csv(config.data_csv_path, encoding='utf8')
    df_all['penalty'] = df_all['penalty'] - 1
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

def train_word2vec(texts):
    '''
    :param texts:  iterable can be simply a list of lists of tokens
        格式： [["cat", "say", "meow"], ["dog", "say", "woof"]]
    :return:
    '''
    LOGGER.log('Train Model...')
    w2v = Word2Vec(texts, size=config.w2v_dim, window=5, iter=15, workers=12, seed=config.seed)
    w2v.save(config.model_w2v)
    LOGGER.log('Save done!')

def main():
    documents = get_raw_documents()
    words_list = get_words_list(documents)
    train_word2vec(words_list)

if __name__ == "__main__":
    main()
