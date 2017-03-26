# coding=UTF-8
class FilePathConfig(object):
    file_root_path = "../file/"
    category_pkl_path = file_root_path + "category.pkl"
    category_file_path = file_root_path + "category.txt"
    train_corpus_path = file_root_path + "train.json"
    test_corpus_path = file_root_path + "test.json"
    raw_lexicon_path = file_root_path + "raw_lexicon.txt"
    selected_lexicon_path = file_root_path + "selected_lexicon.txt"
    selected_features_path = file_root_path + "selected_features.txt"
    cache_file_path = file_root_path + "cache.txt"


    file_encodeing = "UTF-8"
    is_need_print_detail = False
    max_num_features = 35000
    train_ratio = 0.8
    test_ratio = 0.2
    # 是否使用二元字词
    is_use_bigram = False
    def __init__(self):
        pass


class ClassifierConfig(object):
    file_root_path = "../file/"
    train_ratio = 0.8
    test_ratio = 0.2
    max_num_features = 35000
    # 是否使用二元字词
    is_use_bigram = False

    def __init__(self):
        pass

    rf_model_with_common_feature = file_root_path + "rf_model.pkl"
    gbdt_model_with_common_feature = file_root_path + "rf_model.pkl"
    svm_model_with_common_feature = file_root_path + "svm_model.pkl"
    boosting_model_with_common_feature = file_root_path + "rf_model.pkl"

    rf_prams = {}
    gbdt_prams = {}
    svm_prams = {}
