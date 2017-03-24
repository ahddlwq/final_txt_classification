# coding=UTF-8
import cPickle
import codecs
from Queue import PriorityQueue

import numpy as np

from config.config import FilePathConfig
from feature_extractor.entity.document import Document
from feature_extractor.entity.document_vector import DocumentVector
from feature_extractor.entity.lexicon import Lexicon
from feature_extractor.entity.term import Term
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
        data = codecs.open(self.config.cache_file_path, 'rb', self.config.file_encodeing, 'ignore')
        # 默认是int64,但是其实用int32就够了，节省内存，int32最大能够到21亿，我们最大只需要几百万
        feature_stat = np.zeros((len(self.lexicon.name_dic), len(category_dic)), dtype=np.int32)
        feature_freq = np.zeros((len(self.lexicon.name_dic), 1), dtype=np.int32)
        class_size = np.zeros((len(category_dic), 1), dtype=np.int32)

        terms = list()
        num_docs_read = 0
        try:
            for line in data:
                splited_line = line.split("\t")
                if not len(splited_line) == 2:
                    print "Error cache error"
                label_id = splited_line[0]
                # 末尾会有回车符
                id_weight_pairs = splited_line[1].strip().split(",")

                for id_weight_pair in id_weight_pairs:
                    term = Term(id_weight_pair.split(":")[0], id_weight_pair.split(":")[1])
                    terms.append(term)

                class_size[label_id] += 1
                for term in terms:
                    feature_stat[term.term_id][label_id] += 1
                    feature_freq[term.term_id] += 1
                if num_docs_read % 5000 == 0:
                    print "sanned", num_docs_read
        except:
            print "Error selection error"
            data.close()

        print "start cal chi_square"
        selected_features_queue = PriorityQueue(self.config.max_num_features + 1)
        for i in range(0, len(self.lexicon.name_dic)):
            word = self.lexicon.get_word(i)
            if word is not None:
                if word.df == 1 | len(word.name) > 50:
                    continue
            chi_sqr = -1
            chi_max = -1
            for j in range(0, len(category_dic)):
                A = feature_stat[i][j]
                B = feature_freq[i] - A
                C = class_size[j] - A
                D = self.num_doc - A - B - C

                fractor_base = (A + C) * (B + D) * (A + B) * (C + D)
                if fractor_base == 0:
                    chi_sqr = 0
                else:
                    # 不用num_docs，因为都一样
                    chi_sqr = (float)((A * D - B * C) * (A * D - B * C)) / fractor_base
                if chi_sqr > chi_max:
                    chi_max = chi_sqr

            term = Term(i, chi_max)
            selected_features_queue.put(term)
            if selected_features_queue.qsize() > self.config.max_num_featuresL:
                selected_features_queue.get()

        self.output_selected_features(selected_features_queue)

        fid_dic = dict(self.config.max_num_features)
        feature_to_sort = list(selected_features_queue.qsize())
        while selected_features_queue.qsize() > 0:
            term = selected_features_queue.get()
            feature_to_sort.append(term)

        sorted(feature_to_sort)
        for i in range(0, len(feature_to_sort)):
            fid_dic[term.term_id] = i

        return fid_dic

    def output_selected_features(self, selected_features_queue):
        selected_features_file = codecs.open(self.config.selected_lexicon_path, 'wb', self.config.file_encodeing,
                                             'ignore')
        while selected_features_queue.qsize() > 0:
            term = selected_features_queue.get()
            selected_features_file.write(self.lexicon.get_word(term.term_id).name + " " + term.weight + "\n")
        selected_features_file.close()



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
            # self.cache_file.write(str(len(terms)) + '\t')
            if len(terms) > self.longest_length_doc:
                self.longest_length_doc = len(terms)
            for term in terms:
                self.cache_file.write(str(term.term_id) + ":")
                self.cache_file.write(str(term.weight) + ",")
            self.cache_file.write('\n')
        except:
            print "Error write cache error"

        self.num_doc += 0

    # 添加多篇文档，循环调用添加单篇文档
    def add_documents(self, raw_documents):
        for raw_document in raw_documents:
            self.add_document(raw_document)
        # 添加完后，关闭文件
        if self.cache_file is not None:
            print "close"
            self.cache_file.close()

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
