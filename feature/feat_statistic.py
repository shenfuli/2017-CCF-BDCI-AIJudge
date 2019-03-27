# -*- encoding:utf-8 -*-
import sys
sys.path.append("D:/ML_Study/2017-CCF-BDCI-AIJudge")
import re
import numpy as np
import pandas as pd
from config.db_config import Config
from utils import LOGGER
import codecs

config = Config()

def load_data(data_path):
    '''
    加载数据
    :param data_path:
    :return:
    '''
    df_tr = []
    LOGGER.log('For train.txt:')
    for i, line in enumerate(codecs.open(data_path, encoding='utf-8')):
        if i % 1000 == 1:
            LOGGER.log('iter = %d' % i)
        segs = line.split('\t')
        row = {}
        row['id'] = segs[0]
        row['raw_content'] = segs[1].strip()
        df_tr.append(row)
    data_df = pd.DataFrame(df_tr)
    return data_df


if __name__ == "__main__":
    train_df = load_data(config.train_data_path)
    test_df = load_data(config.test_data_path)
    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    amt_list = []
    for i, rows in df.iterrows():
        if i % 1000 == 1:
            LOGGER.log('iter = %d' % i)
        id = rows["id"]
        raw_document = rows['raw_content']
        # 规则1
        row_amount_list = re.findall(u'(\d*\.?\d+)[元]', raw_document)
        for amount in row_amount_list:
            amt_list.append([id, float(amount)])
        # 规则2
        row_amount_list = re.findall(u'(\d*\.?\d+)[万元]', raw_document)
        for amount in row_amount_list:
            amt_list.append([id, float(amount) * 10000])

    # 提取amount 特征
    amt_df = pd.DataFrame(amt_list, columns=["id", "amt"])
    feat_amt = amt_df.groupby(by="id")['amt'].agg([sum, max, min, np.ptp, np.mean, np.std]).reset_index()
    df_merge = df.merge(feat_amt, left_on='id', right_on='id', how='left')
    feat_amt_result = df_merge.drop(columns=['id', 'raw_content'], axis=1)
    feat_amt_result = feat_amt_result.fillna(0)
    feat_amt_result.columns = ['amt_{0}'.format(col) for col in feat_amt_result.columns]

    # 数据保存
    feat_amt_result.to_csv(config.feat_amt, index=None)
