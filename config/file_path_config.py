class FilePathConfig(object):
    file_root_path = "../file/"
    category_pkl_path = file_root_path + "category.pkl"
    category_file_path = file_root_path + "category.txt"
    train_corpus_path = file_root_path + "train.json"
    test_corpus_path = file_root_path + "test.json"

    train_ratio = 0.8
    test_ratio = 0.2
    max_num_features = 35000
    file_encodeing = "UTF-8"
    is_need_print_detail = False

    def __init__(self):
        pass
