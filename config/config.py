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

    sparse_feature_mat_path = file_root_path + "sparse_feature_mat.txt"

    lexicon_pkl_path = file_root_path + "lexicon.pkl"

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

    # 分类器代号
    rf_name = "rf"
    gbdt_name = "gbdt"
    svm_name = "svm"
    rf_prams = {}
    gbdt_prams = {}
    svm_prams = {}

    # 当前系统是使用boosting，还是单模型进行训练和测试
    is_single_model = True
    cur_single_model = rf_name

    # 现在需要进行boosting的分类器集合
    using_classifiers = [rf_name, gbdt_name, svm_name]

    rf_model_with_common_feature = file_root_path + "rf_model.pkl"
    gbdt_model_with_common_feature = file_root_path + "rf_model.pkl"
    svm_model_with_common_feature = file_root_path + "svm_model.pkl"

    classifier_dic = {rf_name: rf_model_with_common_feature,
                      gbdt_name: gbdt_model_with_common_feature,
                      svm_name: svm_model_with_common_feature}

    classifier_weight_dic = {rf_name: 1,
                             gbdt_name: 1,
                             svm_name: 1}

    boosting_model_with_common_feature = file_root_path + "rf_model.pkl"

    boosting_weight_dic_with_common_feature = file_root_path + "boosting_weight_dic.pkl"

    def __init__(self):
        pass
