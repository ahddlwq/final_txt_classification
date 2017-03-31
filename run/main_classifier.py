# coding=UTF-8
import sys

sys.path.append("../")
import cPickle
import codecs
import os
from sklearn.datasets import load_svmlight_file

from config.config import FilePathConfig, ClassifierConfig
from evaluation.test_result import TestResult
from feature_extractor.entity.document import Document
from feature_extractor.entity.document_vector import DocumentVector
from feature_extractor.entity.lexicon import Lexicon
from feature_extractor.entity.termweight import TfOnlyTermWeighter, TfIdfWighter
from feature_extractor.feature_filter.speech_filter import SpeechFilter
from feature_extractor.feature_filter.stop_word_filter import StopWordFilter
from feature_extractor.feature_selection_functions.chi_square import ChiSquare
from feature_extractor.feature_selection_functions.informantion_gain import InformationGain
from misc.util import Util
from model.rf_classifier import RFClassifier


class MainClassifier(object):
    def __init__(self):
        # 主要都是直接从配置文件里读取
        # 加载类别与id的映射字典
        self.category_dic = self.load_category_dic_from_pkl()
        # 加载词典
        self.lexicon = self.load_lexicon()
        # 加载特征降维方法
        self.select_function = self.get_selection_funtion()
        # 将训练转换成文档向量的工具
        self.training_vector_builder = DocumentVector(TfOnlyTermWeighter(self.lexicon))
        # 将测试集转换成文档向量的工具
        self.test_vector_builder = DocumentVector(TfIdfWighter(self.lexicon))
        # 类别数量
        self.num_categories = len(self.category_dic)
        # 缓存文件
        self.cache_file = None
        # 这次要使用的分类器
        self.abstract_classifier = None
        # 文章最长的长度
        self.longest_length_doc = 0
        # 文章的总数量
        self.num_doc = 0


    # -----------------------------------------------------------------------------------------------------------------
    # 训练前的准备，构造词典，特征降维，准备训练数据
    def construct_lexicon(self, train_corpus_path):
        # 从原始语料中转换成稀疏矩阵，保存在cache中，同时会在lexicon里保存下所有的id_dic和name_dic
        self.add_documents_from_file(train_corpus_path)
        # 进行特征降维,返回一个保存了所有特征id的优先队列
        selected_features_queue = self.select_function.feature_select(self.lexicon, self.num_categories, self.num_doc)

        # 获取选择后的特征对应的id，并保存下所有的特征，这里之所以两个功能放在一起做，是由于优先队列出队的不可逆性
        fid_dic = self.output_selected_features_and_get_fid_dic(selected_features_queue)

        # 根据选择后的特征，重新调整词典id的映射关系
        self.lexicon = self.lexicon.map(fid_dic)
        # 当训练好后，则锁定词典,避免再次修改
        self.lexicon.locked = True
        # 保存下词典映射文件
        self.save_lexicon_into_pkl()
        # 根据新的映射关系，将cache文件转化为权重的稀疏矩阵
        # self.convert_and_save_sparse_feature_mat(fid_dic)


    def output_selected_features_and_get_fid_dic(self, selected_features_queue):
        selected_features_file = codecs.open(FilePathConfig.selected_features_path, 'wb', FilePathConfig.file_encodeing,
                                             'ignore')
        fid_dic = dict()
        feature_to_sort = list()
        while selected_features_queue.qsize() > 0:
            term = selected_features_queue.get()
            feature_to_sort.append(term)
            word = self.lexicon.get_word(term.term_id)
            content = word.name + " " + str(word.df) + " " + str(term.weight)
            selected_features_file.write(content + "\n")

        selected_features_file.close()
        sorted(feature_to_sort)
        for i in range(0, len(feature_to_sort)):
            fid_dic[feature_to_sort[i].term_id] = i
        return fid_dic

    # 添加单篇文档用于构造词典
    def add_document(self, raw_document):
        document = Document(raw_document)

        # 检查类别是否合法
        if document.label not in self.category_dic:
            print "Error category error"

        # 如果cache文件还未打开，则打开
        if self.cache_file is None:
            print "open file"
            self.cache_file = codecs.open(FilePathConfig.cache_file_path, 'wb', FilePathConfig.file_encodeing, 'ignore')

        # 添加词的过滤器
        if not ClassifierConfig.is_use_bigram:
            stop_words_filter = StopWordFilter()
            speech_filter = SpeechFilter()
            document.add_filter(speech_filter).add_filter(stop_words_filter)
        content_words = document.get_content_words_feature()
        self.lexicon.add_document(content_words)
        words = self.lexicon.convert_document(content_words)
        terms = self.training_vector_builder.build(words, False, 0)
        try:
            if len(terms) > self.longest_length_doc:
                self.longest_length_doc = len(terms)

            line_result = str(self.category_dic[document.label]) + FilePathConfig.tab
            for term in terms:
                line_result += (str(term.term_id) + FilePathConfig.colon + str(term.weight))
                line_result += FilePathConfig.space
            self.cache_file.write(line_result.strip() + '\n')
        except:
            print "Error write cache error when add document"

        self.num_doc += 1

    # 添加多篇文档，循环调用添加单篇文档
    def add_documents(self, raw_documents):
        count = 0
        for raw_document in raw_documents:
            if count % 10000 == 0:
                print "加载", count
            count += 1
            self.add_document(raw_document)

        self.close_cache()

    # 从文件添加多篇文档，循环调用添加单篇文档
    def add_documents_from_file(self, raw_documents_file_path):
        raw_documents = codecs.open(raw_documents_file_path, 'rb', FilePathConfig.file_encodeing, 'ignore')
        self.add_documents(raw_documents)
        raw_documents.close()

    # -----------------------------------------------------------------------------------------------------------------
    # 分类相关
    # 分类单篇文档，返回多个结果
    def classify_document_top_k(self, raw_document, k):
        document = Document(raw_document)
        return self.abstract_classifier.classify_top_k(document, k)

    # 分类多篇文档，循环调用分类单篇文档，返回多个结果
    def classify_documents_top_k(self, raw_documents, k):
        # classify_results = []
        # for raw_document in raw_documents:
        #     classify_results.append(self.classify_document_top_k(raw_document, k))
        # return classify_results
        return self.abstract_classifier.classify_top_k(raw_documents, 1)

    # 从文件分类多篇文档，循环调用分类单篇文档，返回多个结果
    def classify_documents_top_k_from_file(self, raw_documents_file_path, k):
        raw_documents = codecs.open(raw_documents_file_path, 'rb', FilePathConfig.file_encodeing, 'ignore')
        classify_results = self.classify_documents_top_k(raw_documents, k)
        raw_documents.close()
        return classify_results

    # 分类单篇文档,只返回一个结果
    def classify_document(self, raw_document):
        return self.classify_document_top_k(raw_document, 1)

    # 分类多篇文档，循环调用分类单篇文档,只返回一个结果
    def classify_documents(self, raw_documents):
        return self.classify_documents_top_k(raw_documents, 1)

    # 从文件分类多篇文档，循环调用分类单篇文档,只返回一个结果
    def classify_documents_from_file(self, raw_documents_file_path):
        return self.classify_documents_top_k_from_file(raw_documents_file_path, 1)

    # -----------------------------------------------------------------------------------------------------------------
    # 训练和评测相关
    # 打印分类结果与评测结果
    def print_classify_result(self, predicted_class, raw_class_label):
        labels = sorted(self.category_dic.iteritems(), key=lambda key_value: key_value[1])
        labels = [labels_id_pair[0] for labels_id_pair in labels]
        test_result = TestResult(predicted_class, raw_class_label, labels)
        test_result.print_report()

    def train(self, train_corpus_path):
        print "train"
        self.load_model()
        train_feature_mat, label_vec = self.corpus_to_feature_vec(train_corpus_path,
                                                                  FilePathConfig.train_feature_mat_path)
        self.abstract_classifier.train(train_feature_mat, label_vec)
        pass

    def test(self, test_corpus_path):
        print "test"
        self.load_model()
        test_sparse_mat, label_vec = self.corpus_to_feature_vec(test_corpus_path, FilePathConfig.test_feature_mat_path)
        predicted_class = self.classify_documents(test_sparse_mat)
        print predicted_class
        print label_vec
        self.print_classify_result(predicted_class, label_vec)

    def get_libsvm_data(self, path):
        data = load_svmlight_file(path)
        return data[0], data[1]

    # 将传进来的批量json转换为可用于分类的特征向量矩阵,或者特征向量加原来的分类标签
    def corpus_to_feature_vec(self, corpus_path, result_path):
        if (os.path.isfile(result_path)):
            print "loading"
            return self.get_libsvm_data(result_path)
        jsons = codecs.open(corpus_path, 'rb', FilePathConfig.file_encodeing, 'ignore')
        test_sparse_mat = codecs.open(result_path, 'wb', FilePathConfig.file_encodeing, 'ignore')
        count = 0
        for json in jsons:
            count += 1
            if count % 10000 == 0:
                print "add", count
            document = Document(json)
            label_id = self.category_dic[document.label]
            content_words = document.get_content_words_feature()
            doc_len = len(content_words)

            words = self.lexicon.convert_document(content_words)
            # terms = self.test_vector_builder.build(words, normalized=True)
            terms = self.test_vector_builder.build(words, True, doc_len)

            test_sparse_mat.write(str(label_id))
            terms.sort(cmp=lambda x, y: cmp(x.term_id, y.term_id))
            for term in terms:
                test_sparse_mat.write(" " + str(term.term_id) + ":" + str(term.weight))
            test_sparse_mat.write("\n")

        jsons.close()
        test_sparse_mat.close()
        return self.get_libsvm_data(result_path)

    def load_model(self):
        if ClassifierConfig.is_single_model:
            self.abstract_classifier = RFClassifier()
        else:
            print "None"
            return None

    def load_lexicon(self, lexicon_path=None):
        if lexicon_path is None:
            if os.path.isfile(FilePathConfig.lexicon_pkl_path):
                print "load lexicon"
                return Util.load_object_from_pkl(FilePathConfig.lexicon_pkl_path)
            else:
                return Lexicon()
        else:
            return Util.load_object_from_pkl(FilePathConfig.lexicon_pkl_path)

    # 从文件加载字典对象
    def load_lexicon_from_pkl(self):
        return Util.load_object_from_pkl(FilePathConfig.lexicon_pkl_path)


    # 加载类别与编号字典
    # 返回内容类似为{"时政":1,"军事":2,……}
    def load_category_dic_from_pkl(self):
        return Util.load_object_from_pkl(FilePathConfig.category_pkl_path)

    def save_lexicon_into_pkl(self):
        cPickle.dump(self.lexicon, open(FilePathConfig.lexicon_pkl_path, 'wb'))

    # 调整boosting的权重　
    def adapt_boosting_weight(self):
        pass

    # 选择是哪种特征降维方式
    def get_selection_funtion(self):
        if ClassifierConfig.cur_selection_function == ClassifierConfig.chi_square:
            return ChiSquare()
        else:
            return InformationGain()

    def close_cache(self):
        # 在需要的时候关闭cache文件
        if self.cache_file is not None:
            print "close cache"
            self.cache_file.close()

if __name__ == '__main__':
    # 训练和评测阶段，这里把所有可能需要自定义的参数全部都移到配置文件里了，如果需要也可以换成传参调用的形式
    # 需要外面传进来的参数只有训练集的位置和验证集的位置
    mainClassifier = MainClassifier()
    print mainClassifier.lexicon.locked
    # 根据原始语料进行语料预处理（切词、过滤、特征降维）
    # mainClassifier.construct_lexicon(FilePathConfig.total_corpus_path)
    # 训练
    mainClassifier.train(FilePathConfig.train_corpus_path)
    # 测试
    mainClassifier.test(FilePathConfig.test_corpus_path)
    # ----------------------------------------------------------------------------------------------------
    # 对外来的数据进行分类
    # mainClassifier = MainClassifier()
    # mainClassifier.load_model()
    # # 需要分类的数据
    # raw_document = "{json}"
    # # 只返回单分类
    # classify_result = mainClassifier.classify_document(raw_document)
    # # 需要返回的类别数量
    # k = 3
    # # 返回多个分类和其概率
    # classify_results = mainClassifier.classify_document_top_k(raw_document, k)
