class Config:
    stop_words_path = "/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge/data/input/stopwords.txt"
    train_data_path = "/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge/data/input/train_small.txt"
    test_data_path = "/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge/data/input/test.txt"
    data_csv_path = "/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge/data/output/corpus/data1.csv"
    data_label_path = "/Users/zhengwenjie/AI/work/ML_3/2017-CCF-BDCI-AIJudge/data/output/corpus/data_label.txt"
    cv_train_num = 100000  # 用于交叉验证
    train_num = 120000
    test_num = 90000
    w2v_dim = 300
    seed = 2017

    feat_tfidf_gnb_prob="/output/feature/tfidf/gnb_prob_12w.csv"
    aab=""