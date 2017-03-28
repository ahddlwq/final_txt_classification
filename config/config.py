# coding=UTF-8
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC


class FilePathConfig(object):
    file_root_path = "../file/"
    category_pkl_path = file_root_path + "category.pkl"
    category_file_path = file_root_path + "category.txt"
    train_corpus_path = file_root_path + "train.json"
    test_corpus_path = file_root_path + "train.json"
    raw_lexicon_path = file_root_path + "raw_lexicon.txt"
    selected_lexicon_path = file_root_path + "selected_lexicon.txt"
    selected_features_path = file_root_path + "selected_features.txt"
    cache_file_path = file_root_path + "cache.txt"

    sparse_feature_mat_path = file_root_path + "sparse_feature_mat.txt"

    lexicon_pkl_path = file_root_path + "lexicon.pkl"

    file_encodeing = "UTF-8"
    is_need_print_detail = False

    def __init__(self):
        pass


class ClassifierConfig(object):
    file_root_path = "../file/"
    train_ratio = 0.8
    test_ratio = 0.2
    max_num_features = 35000
    # 是否使用二元字词
    is_use_bigram = False

    # 分类器代号
    rf_name = "rf"
    gbdt_name = "gbdt"
    svm_name = "svm"
    rf_prams = {}
    gbdt_prams = {}
    svm_prams = {}

    # 当前系统是使用boosting，还是单模型进行训练和测试
    is_single_model = True
    cur_single_model = svm_name

    # 现在需要进行boosting的分类器集合
    boosting_using_classifiers = [rf_name, gbdt_name, svm_name]

    rf_model_path = file_root_path + "rf_model.pkl"
    gbdt_model_path = file_root_path + "gbdt_model.pkl"
    svm_model_path = file_root_path + "svm_model.pkl"
    boosting_model_path = file_root_path + "boosting_model.pkl"

    classifier_path_dic = {rf_name: rf_model_path,
                           gbdt_name: gbdt_model_path,
                           svm_name: svm_model_path}

    classifier_pram_dic = {rf_name: rf_prams,
                           gbdt_name: gbdt_prams,
                           svm_name: svm_prams}

    classifier_init_dic = {rf_name: RandomForestClassifier(classifier_pram_dic[rf_name]),
                           gbdt_name: GradientBoostingClassifier(classifier_pram_dic[gbdt_name]),
                           svm_name: LinearSVC(classifier_pram_dic[svm_name])}

    classifier_weight_dic = {rf_name: 1,
                             gbdt_name: 1,
                             svm_name: 1}

    boosting_weight_dic = file_root_path + "boosting_weight_dic.pkl"

    # 降维方式
    chi_square = "chi_square"
    information_gain = "information_gain"
    cur_selection_function = chi_square

    def __init__(self):
        pass
