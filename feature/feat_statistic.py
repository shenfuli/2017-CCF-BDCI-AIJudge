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
        row_amount_list = re.findall(u'(\d*\.?\d+)[元]', raw_document)
        for amount in row_amount_list:
            amt_list.append([id,amount])
     amt_df = pd.DataFrame(amt_list, columns=["id", "amt"])


