# -*- encoding:utf-8 -*-
__author = "shenfuli"

import sys

sys.path.append("D:/ML_Study/2017-CCF-BDCI-AIJudge")
import pandas as pd
from config.db_config import Config
import numpy as np
from utils import LOGGER

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def micro_avg_f1(y_true, y_pred):
    '''
    分类 评估函数
    :param y_true: 样本实际类别
    :param y_pred: 预测类别
    :return:
    '''
    return f1_score(y_true, y_pred, average='micro')


def load_data(data_path):
    '''
    加载数据
    :param data_path:
    :return:
    '''
    train_df = pd.read_csv(data_path, encoding='utf-8', sep=',')
    train_df = train_df.dropna()
    train_df['penalty'] = train_df['penalty'].astype(int)
    return train_df


def get_words_list(raw_documents):
    '''
        获取tfidf 数值化的输入数据
    :return: 数据类型为列表，其中的元素也为列表  [[word1 word2.....],[word11 word12....]]
    '''
    words_list = []
    for document in raw_documents:
        words_list.append(document.split())
    return words_list


def words_list2tfidf_feature(raw_documents):
    words_list = get_words_list(raw_documents)
    ## 1. TfidfVectorizer模型
    '''
    调用sklearn.feature_extraction.text库的TfidfVectorizer方法实例化模型对象。
    TfidfVectorizer方法需要4个参数。
    第1个参数是分词结果，数据类型为列表，其中的元素也为列表；
    第2个关键字参数stop_words是停顿词，数据类型为列表；
    第3个关键字参数min_df是词频低于此值则忽略，数据类型为int或float;
    第4个关键字参数max_df是词频高于此值则忽略，数据类型为Int或float。
    查看TfidfVectorizer方法的更多参数用法，
    官方文档链接：http://sklearn.apachecn.org/cn/0.19.0/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    '''
    tfidf = TfidfVectorizer(words_list, min_df=3, max_df=0.95)
    '''
    第1行代码查看向量化的维数，即特征的维数；
    第2行代码调用TfidfVectorizer对象的fit_transform方法获得特征矩阵赋值给X；
    第3行代码查看特征矩阵的形状。
    '''
    X = tfidf.fit_transform(raw_documents)
    print('词表大小:', len(tfidf.vocabulary_))
    print(X.shape)
    return X


def df_target2label(target):
    '''
    调用sklearn.preprocessing库的LabelEncoder方法对文章分类做标签编码。
    :param target:
    :return:
    '''
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(target)
    return y


## 2. lr 特征提取
def tfidf_clf_features(clf, X_train, X_test, y_train, y_test, num_class, seed=2019):
    # 初始化stack ： 通过模型提取特征矩阵的数值
    print("dataset size X_train={0},X_test={1},dim={2}".format(X_train.shape[0], X_test.shape[0], X_test.shape[1]))
    stack = np.zeros((X_train.shape[0], num_class))
    stack_te = np.zeros((X_test.shape[0], num_class))
    # 1. 交叉验证- 特征得分平均
    k = 5
    print("cross_validate k={0}".format(k))
    score_va = 0
    score_te = 0
    skf = StratifiedKFold(n_splits=k, random_state=seed)
    for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        print("cross_validate.....i={0}".format(i + 1))
        train_data, val_data = X_train[train_index], X_train[val_index]
        train_y, val_y = y_train[train_index], y_train[val_index]
        # clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        clf.fit(train_data, train_y)
        y_pred_val = clf.predict(val_data)
        val_accuracy = micro_avg_f1(val_y, y_pred_val)
        test_accuacy = micro_avg_f1(y_test, clf.predict(X_test))
        LOGGER.log('val_dataset accuacy:%f' % val_accuracy)
        LOGGER.log('test_dataset accuacy:%f' % test_accuacy)
        score_va += val_accuracy
        score_te += test_accuacy
        y_pred_val_prob = clf.predict_proba(val_data)
        y_pred_te_prob = clf.predict_proba(X_test)
        stack[val_index] += y_pred_val_prob
        stack_te += y_pred_te_prob
    score_va /= k
    score_te /= k
    print("cross_validate success. print avg acc ...")
    LOGGER.log('val_dataset avg acc:%f' % score_va)
    LOGGER.log('test_dataset avg acc:%f' % score_te)
    # 2. lr... 提取的特征存储
    stack_all = np.vstack([stack / k, stack_te / k])
    return stack_all


def main():
    config = Config()
    # 1. 提取tfidf 格式特征 X,y
    train_df = load_data(config.data_csv_path)
    count = len(train_df)
    print("data_size={0}".format(count))
    raw_documents = train_df['content']
    X = words_list2tfidf_feature(raw_documents)
    y = df_target2label(train_df['penalty'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    num_class = len(np.unique(y))
    print("num_class={0}".format(num_class))
    # 2. 提取特征
    # 2.1 lr 提取特征
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    stack_all = tfidf_clf_features(clf, X_train, X_test, y_train, y_test, num_class)
    df_stack = pd.DataFrame(index=range(count))
    for i in range(stack_all.shape[1]):
        df_stack['tfidf_lr_{}'.format(i)] = stack_all[:, i]
    df_stack.to_csv(config.feat_tfidf_lr_prob, index=None, encoding='utf8')

    # 2.2 BernoulliNB 提取特征
    clf = BernoulliNB()
    stack_all = tfidf_clf_features(clf, X_train, X_test, y_train, y_test, num_class)
    df_stack = pd.DataFrame(index=range(count))
    for i in range(stack_all.shape[1]):
        df_stack['tfidf_bnb_{}'.format(i)] = stack_all[:, i]
    df_stack.to_csv(config.feat_tfidf_bnb_prob, index=None, encoding='utf8')

    # 2.3 BernoulliNB 提取特征
    clf = MultinomialNB()
    stack_all = tfidf_clf_features(clf, X_train, X_test, y_train, y_test, num_class)
    df_stack = pd.DataFrame(index=range(count))
    for i in range(stack_all.shape[1]):
        df_stack['tfidf_mnb_{}'.format(i)] = stack_all[:, i]
    df_stack.to_csv(config.feat_tfidf_mnb_prob, index=None, encoding='utf8')


if __name__ == "__main__":
    main()
