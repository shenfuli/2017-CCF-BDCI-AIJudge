# -*- encoding:utf-8 -*-
import sys
sys.path.append("/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge")

import codecs
from collections import namedtuple

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from utils import LOGGER
from config.db_config import Config

config = Config()

def write_d2v_data_path(content_list):
    doc_f = codecs.open(config.feat_d2v, encoding='utf-8', mode='w')
    for i, content in enumerate(content_list):
        if i % 10000 == 0:
            LOGGER.log('iter=%d' % i)
        doc_f.write(u'_*{} {}\n'.format(i, content))
    doc_f.close()

SentimentDocument = namedtuple('SentimentDocument', 'words tags')
class Doc_list(object):
    def __init__(self, f):
        self.f = f
    def __iter__(self):
        for i,line in enumerate(codecs.open(self.f,encoding='utf8')):
            words = line.strip().split(' ')
            tags = [int(words[0][2:])]
            words = words[1:]
            yield SentimentDocument(words,tags)

def main():
    data_df = pd.read_csv(config.data_csv_path, encoding='utf-8')
    y = data_df['penalty'] - 1
    rows = data_df.shape[0]
    # 生成d2v 初始文件数据
    write_d2v_data_path(data_df['content'])
    doc_list = Doc_list(config.feat_d2v)

    '''
        dm:
        int {1,0} : `dm=1` indicates 'distributed memory' (PV-DM) else
        `distributed bag of words` (PV-DBOW) is used.""
    '''
    print("distributed bag of words ..........")
    d2v = Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=3, window=30, sample=1e-5, workers=8, alpha=0.025,min_alpha=0.025)
    d2v.build_vocab(doc_list)

    for i in range(5):
        LOGGER.log('pass:' + str(i))
        d2v.train(doc_list, total_examples=d2v.corpus_count, epochs=d2v.iter)
        X_d2v = np.array([d2v.docvecs[i] for i in range(rows)])
        '''
            Evaluate a score by cross-validation
            Array of scores of the estimator for each run of the cross validation
            
            scores：返回cv＝6 每次的得分
        '''
        scores = cross_val_score(LogisticRegression(C=3.0),X_d2v,y,cv=5) #ross_val_score(lasso, X, y)
        LOGGER.log('dbow: ' + str(scores) + ' ' + str(np.mean(scores)))
    d2v.save(config.model_d2v_dbow)
    LOGGER.log('Save done!')

    print("distributed memory..........")
    d2v = Doc2Vec(dm=1, size=100, negative=5, hs=0, min_count=3, window=30, sample=1e-5, workers=8, alpha=0.025,
                  min_alpha=0.025)
    d2v.build_vocab(doc_list)
    for i in range(5):
        LOGGER.log('pass:' + str(i))
        d2v.train(doc_list, total_examples=d2v.corpus_count, epochs=d2v.iter)
        X_d2v = np.array([d2v.docvecs[i] for i in range(rows)])
        scores = cross_val_score(LogisticRegression(C=3.0), X_d2v, y, cv=5)  # ross_val_score(lasso, X, y)
        LOGGER.log('dm: ' + str(scores) + ' ' + str(np.mean(scores)))
    d2v.save(config.model_d2v_dm)
    LOGGER.log('Save done!')

if __name__=="__main__":
    main()