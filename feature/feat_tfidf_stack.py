# -*- encoding:utf-8 -*-
__author = "shenfuli"

import pandas as pd
from config.db_config import Config
import numpy as np
from utils import LOGGER

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import codecs

#https://blog.csdn.net/xiaosa_kun/article/details/84868437
def micro_avg_f1(y_true, y_pred):
    '''
    分类 评估函数
    :param y_true: 样本实际类别
    :param y_pred: 预测类别
    :return:
    '''
    return f1_score(y_true, y_pred, average='micro')


def load_data():
    config = Config()

    documents = []
    for document in codecs.open(config.data_label_path, "r", encoding="utf-8"):
        document = document.strip()
        data = document.split("\t")
        words = data[0]
        label = data[1]
        documents.append((words.split(" "), label))

    # 1. 获取训练和测试数据
    X, y = zip(*documents)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234, test_size=0.3)
    # print(X_train[0:2])
    # print(y_train[0:2])
    # print("train data size = {0}".format(len(y_train)))
    # print("test  data size = {0}".format(len(y_test)))

    # 2.tf-idf 特征转换
    tfidf = TfidfVectorizer(X,min_df=3,max_df=0.95,sublinear_tf=True)
    X2 = tfidf.fit_transform(X)
    print(X2)
if __name__ == "__main__":
    load_data()
