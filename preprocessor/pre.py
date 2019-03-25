# -*- encoding:utf-8 -*-
__author = "shenfuli"

'''

    数据预处理
    1. 输入数据格式： 训练的文本数据，格式

    比赛中提供的训练数据为多行文本，每一行分为四列，使用\t分割，第一列为文档ID, 第二列为案件事实描述，第三列为罚金额度类别，第四列为对应的法律条文编号序列，其中法律条文编号序列是有”,”号分割。
    数据样例如下，以表格形式进行表示：

    1
    被告人高某明知罂粟为毒品，于2012年11月份将捡拾的罂粟籽种植在其院内。2013年4月8日，公安机关民警在巡逻中发现被告人高某家中种植的罂粟后遂将罂粟植株予以铲除、扣押。经现场清点，高某种植的罂粟植株共计622株。
    1
    351,67,72,73


    2. 输出数据格式
    生成一个csv的文件，合并train.txt,test.txt ,其中test.txt 中仅有id content 两列，其中： content 部分是 通过空格分开的分词数据
    id  content penalty  laws
    －－－－－－－－－－－－－－－

'''
import codecs
import jieba
import jieba.analyse
import jieba.posseg
import pandas as pd
from utils import LOGGER
from config.db_config import Config

def split_word(text, stopwords):
    '''
    定义分词函数
    :param text:
    :param stopwords:
    :return:
    '''
    word_list = jieba.cut(text)
    seg = [word.strip() for word in word_list if len(word) > 1 and word not in stopwords]
    result = " ".join(seg)
    return result


def load_stop_words(stop_words_path):
    '''
    加载停用词
    :param stop_words_path:
    :return:
    '''
    stopwords = {}
    for line in codecs.open(stop_words_path, 'r', 'utf-8'):
        stopwords[line.rstrip()] = 1
    return stopwords


def load_data(data_path, stopwords):
    '''
    加载训练／测试数据
    :param data_path:
    :param stopwords:
    :return:  返回  id  content penalty  laws 的数据结构，其中： test数据仅 id content 格式的结构
    '''
    data_list = []
    for i, line in enumerate(codecs.open(data_path,encoding='utf-8')):
        if i % 1000 == 1:
            LOGGER.log('iter = %d' % i)
        segs = line.split('\t')
        row = {}
        row['id'] = segs[0]
        row['content'] = split_word(segs[1].strip(), stopwords)
        row['penalty'] = segs[2] if len(segs) >= 3 else ""
        row['laws'] = segs[3].strip() if len(segs) >= 4 else ""
        data_list.append(row)
    return data_list


def main():
    config = Config()
    stopwords = load_stop_words(config.stop_words_path)
    train_data_list = load_data(config.train_data_path, stopwords)
    test_data_list = load_data(config.test_data_path,stopwords)
    train_data_df = pd.DataFrame(train_data_list, columns=["id", "content", "laws", "penalty"])
    test_data_df = pd.DataFrame(test_data_list,columns=["id", "content", "laws", "penalty"])
    print("train_data rows={},test_data rows={}".format(train_data_df.shape[0], test_data_df.shape[0]))
    print(train_data_df.head())
    print(test_data_df.head())
    data_df = pd.concat([train_data_df,test_data_df])
    data_df.to_csv(config.data_csv_path, index=None)

if __name__ == "__main__":
    main()
