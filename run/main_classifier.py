# coding=UTF-8
import cPickle
import codecs

from config.file_path_config import FilePathConfig
from feature_extractor.entity.document import Document
from feature_extractor.entity.document_vector import DocumentVector
from feature_extractor.entity.lexicon import Lexicon
from feature_extractor.entity.termweight import TfOnlyTermWeighter
from misc.util import Util
from model.svm_classifier import SVMClassifier


class MainClassifier(object):
    def __init__(self):
        self.config = FilePathConfig()
        # 默认为SVM分类器
        self.abstract_classifier = SVMClassifier()
        self.category_dic = self.load_category_dic_from_pkl()
        self.lexicon = Lexicon()
        self.util = Util()
        self.num_categories = len(self.category_dic)
        self.cache_file = None
        self.longest_length_doc = 0
        self.training_vector_builder = DocumentVector(TfOnlyTermWeighter(self.lexicon))
        self.num_doc = 0

    def construct_lexicon(self):
        self.add_documents_from_file(self.config.train_corpus_path)
        self.util.save_collection_into_file(self.config.raw_lexicon_path, self.lexicon.name_dic.keys())
        self.feature_selection()
        pass

    def feature_selection(self):
        print "selection"
        if self.cache_file is not None:
            print "close"
            self.cache_file.close()
        pass

    # 加载类别与编号字典
    # 返回内容类似为{"时政":1,"军事":2,……}
    def load_category_dic_from_pkl(self):
        return cPickle.load(open(self.config.category_pkl_path, 'r'))

    # 将文档转化为可用的内容
    def document_filter(self):
        pass

    # 添加单篇文档
    def add_document(self, raw_document):
        document = Document(raw_document)
        if document.label not in self.category_dic:
            print "Error category error"

        if self.cache_file is None:
            print "open file"
            self.cache_file = codecs.open(self.config.cache_file_path, 'wb', self.config.file_encodeing, 'ignore')

        content_words = document.get_content_words()
        self.lexicon.add_document(content_words)
        words = self.lexicon.convert_document(content_words)
        terms = self.training_vector_builder.build(words, False)
        try:
            self.cache_file.write(str(self.category_dic[document.label]) + '\t')
            self.cache_file.write(str(len(terms)) + '\t')
            if len(terms) > self.longest_length_doc:
                self.longest_length_doc = len(terms)
            for term in terms:
                self.cache_file.write(str(term.term_id) + ":")
                self.cache_file.write(str(term.weight) + " ")
            self.cache_file.write('\n')
        except:
            print "Error write cache error"

        self.num_doc += 0


    # 添加多篇文档，循环调用添加单篇文档
    def add_documents(self, raw_documents):
        for raw_document in raw_documents:
            self.add_document(raw_document)

    # 从文件添加多篇文档，循环调用添加单篇文档
    def add_documents_from_file(self, raw_documents_file_path):
        raw_documents = codecs.open(raw_documents_file_path, 'rb', self.config.file_encodeing, 'ignore')
        self.add_documents(raw_documents)

    # 分类单篇文档
    def classify_document(self, raw_document):
        document = Document(raw_document)
        self.abstract_classifier.classify(document)
        pass

    # 分类多篇文档，循环调用分类单篇文档
    def classify_documents(self, raw_documents):
        for raw_document in raw_documents:
            self.classify_document(raw_document)

    # 从文件分类多篇文档，循环调用分类单篇文档
    def classify_documents_from_file(self, raw_documents_file_path):
        raw_documents = codecs.open(raw_documents_file_path, 'rb', self.config.file_encodeing, 'ignore')
        self.classify_documents(raw_documents)

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
    category_dic = mainClassifier.construct_lexicon()
