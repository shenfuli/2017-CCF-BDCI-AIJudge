class Config:
    stop_words_path = "/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge/data/input/stopwords.txt"
    train_data_path = "/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge/data/input/train_small.txt"
    test_data_path = "/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge/data/input/test_small.txt"
    data_csv_path = "/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge/data/output/corpus/data.csv"
    data_label_path = "/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge/data/output/corpus/data_label.txt"
    cv_train_num = 100000  # 用于交叉验证
    train_num = 120000
    test_num = 90000
    seed = 2017
    # tfidf-lr 特征提取结果
    feat_tfidf_lr_prob = '/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge/data/output/feature/tfidf/lr_prob_12w.csv'
    feat_tfidf_bnb_prob = '/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge/data/output/feature/tfidf/bnb_prob_12w.csv'
    feat_tfidf_mnb_prob = '/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge/data/output/feature/tfidf/mnb_prod_12w.csv'

    # word2vec 特征提取
    model_w2v="/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge/data/output/model/w2v_12w.model"
    w2v_dim = 300
    feat_w2v = "/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge/data/output/feature/w2v/w2v_12w.csv"
    feat_w2v_avg = "D:/ML_Study/2017-CCF-BDCI-AIJudge/data/output/feature/w2v/w2v_avg_12w.csv"

    # 统计类-金额  特征
    feat_amt = "/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge/data/output/feature/amt/amt_21w.csv"

    feat_d2v="/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge/data/output/corpus/doc_for_d2v_12w.txt"

    # doc2vec
    model_d2v_dbow = "/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge/data/output/model/dbow_d2v_12w.model"
    model_d2v_dm = "/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge/data/output/model/dm_d2v_12w.model"


    aab=""