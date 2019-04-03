# -*- encoding:utf-8 -*-
import sys

sys.path.append("/home/julyedu_41116/2017-CCF-BDCI-AIJudge")

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from utils import LOGGER
from config.db_config import Config
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

import warnings

warnings.filterwarnings('ignore')
config = Config()


def micro_avg_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


def main():
    # 加载数据
    print("1.load data......")
    data_df = pd.read_csv(config.data_csv_path, encoding='utf-8')
    y = data_df['penalty'] - 1
    rows = data_df.shape[0]
    num_class = len(np.unique(y))
    print(rows, num_class)
    print(data_df.head())

    print("2. load doc2vec model.init word vectors......")
    pre_train_model = Doc2Vec.load(config.model_d2v_dbow)
    X = np.array([pre_train_model.docvecs[idx] for idx in range(rows)])
    print("dataset size rows={0},cols={1}".format(X.shape[0], X.shape[1]))

    print("3. split dataset train and test,init stack matrix ......")
    TR = int(0.7 * rows)
    X_train, X_test = X[:TR], X[TR:]
    y_train, y_test = y[0:TR], y[TR:]
    print("dataset size X_train={0},X_test={1},dim={2}".format(X_train.shape[0], X_test.shape[0], X_test.shape[1]))
    # 初始化数据矩阵，默认数据为0
    stack = np.zeros((X_train.shape[0], num_class))
    stack_te = np.zeros((X_test.shape[0], num_class))
    print(stack.shape, stack_te.shape)

    # 交叉验证- 特征得分平均
    score_va = 0
    score_te = 0
    k = num_class
    skf = StratifiedKFold(n_splits=k, random_state=config.seed)
    for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        print("cross_validate.....i={0}".format(i + 1))
        train_x, val_x = X_train[train_index], X_train[val_index]
        train_y, val_y = y_train[train_index], y_train[val_index]

        model = Sequential()
        model.add(Dense(300, input_shape=(train_x.shape[1],)))
        model.add(Dropout(0.1))
        model.add(Activation('tanh'))
        model.add(Dense(num_class))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])
        model.fit(train_x, np_utils.to_categorical(train_y, num_class), shuffle=True,
                  batch_size=128, nb_epoch=35,
                  verbose=2, validation_data=(val_x, np_utils.to_categorical(val_y, num_class)))

        val_accuracy = micro_avg_f1(val_y, model.predict_classes(val_x))
        test_accuacy = micro_avg_f1(y_test, model.predict_classes(X_test))
        LOGGER.log('val_dataset accuacy:%f' % val_accuracy)
        LOGGER.log('test_dataset accuacy:%f' % test_accuacy)

        # 特征抽取
        y_pred_proba_va = model.predict_proba(val_x)
        y_pred_proba_te = model.predict_proba(X_test)
        stack[val_index] += y_pred_proba_va
        stack_te += y_pred_proba_te
        # 平均得分
        score_va += val_accuracy
        score_te += test_accuacy

    score_va /= k
    score_te /= k
    print("cross_validate success. print avg acc ...")
    LOGGER.log('val_dataset avg acc:%f' % score_va)
    LOGGER.log('test_dataset avg acc:%f' % score_te)
    stack_all = np.vstack([stack / k, stack_te / k])
    df_stack = pd.DataFrame(index=range(rows))
    for i in range(stack_all.shape[1]):
        df_stack['feat_dbowd2v_{}'.format(i)] = stack_all[:, i]

    print(df_stack.head())
    df_stack.to_csv(config.feat_dbowd2v_prob, index=None, encoding='utf8')
if __name__ == "__main__":
    main()
