# -*- encoding:utf-8 -*-
import sys
sys.path.append("/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge")

import codecs
import subprocess
from collections import namedtuple

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from utils import LOGGER
from config.db_config import Config


config = Config()

def write_d2v_data_path():
    data_df = pd.read_csv(config.data_csv_path, encoding='utf-8')

    doc_f = codecs.open(config.feat_d2v, encoding='utf-8', mode='w')
    for i, content in enumerate(data_df['content']):
        if i % 10000 == 0:
            LOGGER.log('iter=%d' % i)
        doc_f.write(u'_*{} {}\n'.format(i, content))
    doc_f.close()


def main():
    # 生成d2v 初始文件数据
    write_d2v_data_path()


    print("hello world")

if __name__=="__main__":
    main()