# coding=UTF-8
import cPickle
import codecs

from config.config import FilePathConfig
from feature_extractor.entity.document import Document
from feature_extractor.entity.document_vector import DocumentVector
from feature_extractor.entity.lexicon import Lexicon
from feature_extractor.entity.termweight import TfOnlyTermWeighter
from feature_extractor.feature_selection_functions.chi_square import ChiSquare
from misc.util import Util
from model.svm_classifier import SVMClassifier


class MainClassifier(object):
    def __init__(self):
        self.config = FilePathConfig()
        # 默认为SVM分类器
        self.abstract_classifier = SVMClassifier()
        self.category_dic = self.load_category_dic_from_pkl()
        self.num_categories = len(self.category_dic)
        self.lexicon = Lexicon()
        self.util = Util()
        self.cache_file = None
        self.longest_length_doc = 0
        self.training_vector_builder = DocumentVector(TfOnlyTermWeighter(self.lexicon))
        self.test_vector_builder = None
        self.num_doc = 0
        self.select_function = ChiSquare()

    def construct_lexicon(self):
        self.add_documents_from_file(self.config.train_corpus_path)
        self.close_cache()
        self.util.save_collection_into_file(self.config.raw_lexicon_path, self.lexicon.name_dic.keys())
        print self.config.cache_file_path, self.num_categories, self.num_doc, len(self.lexicon.name_dic)
        selected_features_queue = self.select_function.feature_select(self.config.cache_file_path, self.lexicon,
                                                                      self.num_categories, self.num_doc)

        self.output_selected_features(set(selected_features_queue))
        fid_dic = self.get_fid_dic(selected_features_queue)

        # print len(fid_dic), "fid_dic"
        #
        # #根据选择后的特征，重新调整词典
        # self.lexicon = self.lexicon.map(fid_dic)
        #
        # print len(self.lexicon.name_dic)
        # self.util.save_collection_into_file(self.config.selected_lexicon_path,self.lexicon.name_dic.keys())

        # self.lexicon.locked = True
        # self.training_vector_builder = None
        # self.test_vector_builder = DocumentVector(TfIdfWighter(self.lexicon))

    def close_cache(self):
        # 添加完后，关闭文件
        if self.cache_file is not None:
            print "close cache"
            self.cache_file.close()

    def get_fid_dic(self, selected_features_queue):
        fid_dic = dict()
        feature_to_sort = list()
        while selected_features_queue.qsize() > 0:
            term = selected_features_queue.get()
            feature_to_sort.append(term)

        sorted(feature_to_sort)
        for i in range(0, len(feature_to_sort)):
            fid_dic[term.term_id] = i
        return fid_dic

    def output_selected_features(self, selected_features_array):
        selected_features_file = codecs.open(self.config.selected_features_path, 'wb', self.config.file_encodeing,
                                             'ignore')
        for term in selected_features_array:
            content = self.lexicon.get_word(term.term_id).name + " " + str(term.weight)
            selected_features_file.write(content + "\n")

        selected_features_file.close()



    # 加载类别与编号字典
    # 返回内容类似为{"时政":1,"军事":2,……}
    def load_category_dic_from_pkl(self):
        return cPickle.load(open(self.config.category_pkl_path, 'r'))

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
            if len(terms) > self.longest_length_doc:
                self.longest_length_doc = len(terms)

            line_result = str(self.category_dic[document.label]) + '\t'
            for term in terms:
                line_result += (str(term.term_id) + ":" + str(term.weight))
                line_result += ","
            # 去除最后一个逗号
            line_result = line_result[:-1]
            self.cache_file.write(line_result + '\n')
        except:
            print "Error write cache error when add document"

        self.num_doc += 1

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
