# coding=UTF-8
import cPickle

from config.file_path_config import FilePathConfig


class MainClassifier(object):
    def __init__(self):
        self.config = FilePathConfig()

    # 加载类别与编号字典
    def load_category_dic_from_pkl(self):
        category_dic = cPickle.load(open(self.config.category_pkl_path, 'r'))
        return category_dic

    # 将文档转化为可用的内容
    def document_filter(self):
        pass

    # 添加单篇文档
    def add_document(self):
        pass

    # 添加多篇文档，循环调用添加单篇文档
    def add_documents(self):
        pass

    # 分类单篇文档
    def classify_document(self):
        pass

    # 分类多篇文档，循环调用分类单篇文档
    def classify_documents(self):
        pass

    # 打印分类结果
    def print_classify_result(self):
        pass

    # 以随机森林分类器进行分类
    def run_as_rf_classifier(self):
        pass

    # 以GBDT分类器进行分类
    def run_as_gbdt_classifier(self):
        pass

    # 以总的boosting进行分类
    def run_as_boosting_classifier(self):
        pass


if __name__ == '__main__':
    mainClassifier = MainClassifier()
    category_dic = mainClassifier.load_category_dic_from_pkl()
    for key, value in category_dic.items():
        print key, value
