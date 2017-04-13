# coding=UTF-8
from multiprocessing import cpu_count

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC

class FilePathConfig(object):
    file_root_path = "../file/"
    config_root_path = "../config/"
    category_file_path = config_root_path + "category.txt"

    category_pkl_path = file_root_path + "category.pkl"
    stop_words_path = config_root_path + "stop_words.txt"

    total_corpus_path = file_root_path + "total_corpus.json"
    train_corpus_path = file_root_path + "train.json"
    test_corpus_path = file_root_path + "test.json"
    cache_file_path = file_root_path + "cache.txt"
    train_feature_mat_path = file_root_path + "train_sparse_feature_mat.txt"
    test_feature_mat_path = file_root_path + "test_sparse_feature_mat.txt"
    lexicon_pkl_path = file_root_path + "lexicon.pkl"

    raw_news_path = file_root_path + "news.json"
    raw_feature_path = file_root_path + "raw_mat.txt"

    file_encodeing = "UTF-8"
    is_need_print_detail = False

    result_report_path = file_root_path + "result_report%s.pkl"
    log_path = file_root_path + "all.log"
    # 用于分割文件中的内容
    tab = "\t"
    colon = ":"
    space = " "

    raw_lexicon_path = file_root_path + "raw_lexicon.txt"
    selected_lexicon_path = file_root_path + "selected_lexicon.txt"
    selected_features_path = file_root_path + "selected_features.txt"

    def __init__(self):
        pass


class ClassifierConfig(object):
    file_root_path = "../file/"
    train_ratio = 0.8
    test_ratio = 0.2
    max_num_features = 35000
    # 是否使用二元字词
    is_use_bigram = False
    # 获取可用的CPU数量，用于配置分类器，使用四分之三的CPU
    cpu_counts = cpu_count() * 3 / 4
    # 分类器代号
    rf_name = "rf"
    xgb_name = "xgb"
    svm_name = "svm"
    lr_name = "lr"
    gnb_name = "gnb"
    mnb_name = "mnb"
    grid_search_name = "grid"
    boosting_name = "boosting"

    rf_prams = {"n_estimators": 100, "n_jobs": cpu_counts, "random_state": 1, "max_depth": 100, "min_samples_split": 5,
                "min_samples_leaf": 5}
    xgb_prams = {"max_depth": 30, "seed": 1, "nthread": cpu_counts, "silent": False, "n_estimators": 50,
                 "subsample": 0.8}
    svm_prams = {}
    lr_prams = {"random_state": 1, "n_jobs": cpu_counts}
    gnb_prams = {}
    mnb_prams = {}


    rf_grid_search_prams = {"max_depth": range(50, 150, 20)}
    gsearch = GridSearchCV(estimator=RandomForestClassifier(n_estimators=200, oob_score=True,
                                                            random_state=1, n_jobs=cpu_counts, min_samples_leaf=15,
                                                            min_samples_split=15),
                           param_grid=rf_grid_search_prams, iid=False, cv=3)

    # 当前系统是使用boosting，还是单模型进行训练和测试
    is_single_model = True
    is_grid_search = True

    # 用于迭代产生训练数据的分类器
    train_data_claasifiers = [svm_name, lr_name, xgb_name, mnb_name]
    # 能够预测，给出概率的分类器
    can_predict_pro_classifiers = [rf_name, xgb_name, lr_name, gnb_name, mnb_name]

    can_partial_train_predict_classifiers = [gnb_name, mnb_name]

    cur_single_model = mnb_name

    # 现在需要进行boosting的分类器集合
    boosting_using_classifiers = [rf_name, xgb_name, svm_name]

    rf_model_path = file_root_path + "model_" + rf_name + ".pkl"
    xgb_model_path = file_root_path + "model_" + xgb_name + ".pkl"
    svm_model_path = file_root_path + "model_" + svm_name + ".pkl"
    lr_model_path = file_root_path + "model_" + lr_name + ".pkl"
    gnb_model_path = file_root_path + "model_" + gnb_name + ".pkl"
    mnb_model_path = file_root_path + "model_" + mnb_name + ".pkl"
    grid_search_model_path = file_root_path + "model_" + grid_search_name + ".pkl"
    boosting_model_path = file_root_path + "model_" + boosting_name + ".pkl"

    classifier_path_dic = {rf_name: rf_model_path,
                           xgb_name: xgb_model_path,
                           svm_name: svm_model_path,
                           lr_name: lr_model_path,
                           gnb_name: gnb_model_path,
                           grid_search_name: grid_search_model_path,
                           mnb_name: mnb_model_path}

    classifier_pram_dic = {rf_name: rf_prams,
                           xgb_name: xgb_prams,
                           svm_name: svm_prams,
                           lr_name: lr_prams,
                           gnb_name: gnb_prams,
                           mnb_name: mnb_prams}

    classifier_init_dic = {rf_name: RandomForestClassifier(**classifier_pram_dic[rf_name]),
                           xgb_name: xgb.XGBClassifier(**classifier_pram_dic[xgb_name]),
                           svm_name: LinearSVC(**classifier_pram_dic[svm_name]),
                           lr_name: LogisticRegression(**classifier_pram_dic[lr_name]),
                           gnb_name: GaussianNB(**classifier_pram_dic[gnb_name]),
                           grid_search_name: gsearch,
                           mnb_name: MultinomialNB(**classifier_pram_dic[mnb_name])}

    classifier_weight_dic = {lr_name: 1,
                             xgb_name: 1,
                             svm_name: 1}

    boosting_weight_dic = file_root_path + "boosting_weight_dic.pkl"

    # 降维方式
    chi_square = "chi_square"
    information_gain = "information_gain"
    cur_selection_function = chi_square

    def __init__(self):
        pass
