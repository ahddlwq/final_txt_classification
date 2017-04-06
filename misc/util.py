import cPickle
import codecs
import os
import sys

from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file

from config.config import FilePathConfig, ClassifierConfig
from log_tool import LogTool


class Util(object):
    log_tool = LogTool()

    def __init__(self):
        pass

    @staticmethod
    def save_cate_dic_into_pkl():
        cate_file = codecs.open(FilePathConfig.category_file_path, "rb", FilePathConfig.file_encodeing, "ignore")
        cate_dic = {}
        cate_id = 0
        for line in cate_file:
            cate_dic[line.strip()] = cate_id
            cate_id += 1
        Util.save_object_into_pkl(cate_dic, FilePathConfig.category_pkl_path)

    @staticmethod
    def save_object_into_pkl(saved_object, pkl_path):
        cPickle.dump(saved_object, open(pkl_path, 'wb'))

    @staticmethod
    def save_collection_strs_into_file(collection_file_path, contents):
        collection_file = codecs.open(collection_file_path, 'wb', FilePathConfig.file_encodeing, 'ignore')
        lined_contents = '\n'.join(contents)
        collection_file.write(lined_contents)
        collection_file.close()

    @staticmethod
    def load_object_from_pkl(pkl_path):
        return cPickle.load(open(pkl_path, 'r'))

    @staticmethod
    def quit():
        try:
            print "exit"
            sys.exit(0)
        except:
            pass
        finally:
            pass

    @staticmethod
    def is_file(file_path):
        return os.path.isfile(file_path)

    @staticmethod
    def get_libsvm_data(path):
        data = load_svmlight_file(path, n_features=ClassifierConfig.max_num_features)
        return data[0], data[1]

    @staticmethod
    def save_svmlight_file(x, y, path):
        dump_svmlight_file(x, y, path, False)

if __name__ == '__main__':
    util = Util()
    # util.save_cate_dic_into_pkl()
    util.log_tool.log.info("asd2")
