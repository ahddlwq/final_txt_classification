# coding=UTF-8
class FilePathConfig(object):
    file_root_path = "../file/"
    category_pkl_path = file_root_path + "category.pkl"
    category_file_path = file_root_path + "category.txt"
    train_corpus_path = file_root_path + "train.json"
    test_corpus_path = file_root_path + "test.json"
    raw_lexicon_path = file_root_path + "raw_lexicon.txt"
    selected_lexicon_path = file_root_path + "selected_lexicon.txt"
    cache_file_path = file_root_path + "cache.txt"

    train_ratio = 0.8
    test_ratio = 0.2
    max_num_features = 35000
    file_encodeing = "UTF-8"
    is_need_print_detail = False

    # 是否使用二元字词
    is_use_bigram = False

    def __init__(self):
        pass
