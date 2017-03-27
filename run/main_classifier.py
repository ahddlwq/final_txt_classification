# coding=UTF-8
import cPickle
import codecs

from config.config import FilePathConfig, ClassifierConfig
from feature_extractor.entity.document import Document
from feature_extractor.entity.document_vector import DocumentVector
from feature_extractor.entity.lexicon import Lexicon
from feature_extractor.entity.termweight import TfOnlyTermWeighter, TfIdfWighter
from feature_extractor.feature_selection_functions.chi_square import ChiSquare
from feature_extractor.feature_selection_functions.speech_filter import SpeechFilter
from feature_extractor.feature_selection_functions.stop_word_filter import StopWordFilter
from misc.util import Util


class MainClassifier(object):
    def __init__(self):
        self.abstract_classifier = None
        self.category_dic = self.load_category_dic_from_pkl()
        self.num_categories = len(self.category_dic)
        self.lexicon = self.load_lexicon()
        self.util = Util()
        self.cache_file = None
        self.longest_length_doc = 0
        self.training_vector_builder = DocumentVector(TfOnlyTermWeighter(self.lexicon))
        self.test_vector_builder = DocumentVector(TfIdfWighter(self.lexicon))
        self.num_doc = 0
        self.select_function = ChiSquare()
        self.model = None

    # 这一部分主要都是前期模型的训练，准备
    # ---------------------------------------------------------------------------------------------
    def construct_lexicon(self):
        # 从原始语料中转换成稀疏矩阵，保存在cache中，同时会在lexicon里保存下所有的id_dic和name_dic
        self.add_documents_from_file(FilePathConfig.train_corpus_path)
        # 关闭cache
        self.close_cache()
        # #将原始的所有词库保存下来
        # self.util.save_collection_into_file(FilePathConfig.raw_lexicon_path, self.lexicon.name_dic.keys())
        # 进行特征降维,返回一个保存了所有特征id的优先队列
        selected_features_queue = self.select_function.feature_select(self.lexicon, self.num_categories, self.num_doc)

        # 获取选择后的特征对应的id，并保存下所有的特征，这里之所以两个功能一起做，是由于优先队列出队的不可逆性
        fid_dic = self.output_selected_features_and_get_fid_dic(selected_features_queue)

        # 根据选择后的特征，重新调整词典id的映射关系
        self.lexicon = self.lexicon.map(fid_dic)
        # self.lexicon.locked = True
        # 保存下词典映射文件
        self.save_lexicon_into_pkl()

        return fid_dic

    # 根据选取后的特征和原来的cache保存的稀疏矩阵构造出带权重的特征稀疏矩阵
    def get_sparse_feature_mat(self, fid_dic, cache_file_path):

        pass

    def close_cache(self):
        # 添加完后，关闭文件
        if self.cache_file is not None:
            print "close cache"
            self.cache_file.close()

    def output_selected_features_and_get_fid_dic(self, selected_features_queue):
        selected_features_file = codecs.open(FilePathConfig.selected_features_path, 'wb', FilePathConfig.file_encodeing,
                                             'ignore')
        fid_dic = dict()
        feature_to_sort = list()
        while selected_features_queue.qsize() > 0:
            term = selected_features_queue.get()
            feature_to_sort.append(term)
            content = self.lexicon.get_word(term.term_id).name + " " + str(term.weight)
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
        stop_words_filter = StopWordFilter()
        speech_filter = SpeechFilter()
        document.add_filter(speech_filter).add_filter(stop_words_filter)
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
                line_result += " "
            # 去除最后一个逗号
            # line_result = line_result[:-1]
            self.cache_file.write(line_result.strip() + '\n')
        except:
            print "Error write cache error when add document"

        self.num_doc += 1

    # 添加多篇文档，循环调用添加单篇文档
    def add_documents(self, raw_documents):
        for raw_document in raw_documents:
            self.add_document(raw_document)

    # 从文件添加多篇文档，循环调用添加单篇文档
    def add_documents_from_file(self, raw_documents_file_path):
        raw_documents = codecs.open(raw_documents_file_path, 'rb', FilePathConfig.file_encodeing, 'ignore')
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
        raw_documents = codecs.open(raw_documents_file_path, 'rb', FilePathConfig.file_encodeing, 'ignore')
        self.classify_documents(raw_documents)

    # 打印分类结果
    def print_classify_result(self):
        pass

    def train(self, feature_mat, label_vec):
        # self.construct_lexicon()

        self.model = self.abstract_classifier.train(feature_mat, label_vec)
        pass

    def get_feature_mat_from_cache_file(self):

        pass

    def test(self, feature_mat, label_vec):
        self.print_classify_result()
        pass

    # 通用的对常见特征进行分类
    def run_classifier_with_common_feature(self, model, feature_mat):
        classify_results = []
        for feature_vec in feature_mat:
            classify_result = model.classify(feature_vec)
        classify_results.append(classify_result)
        return classify_results

    # 以总的boosting对常见的特征进行分类
    def run_as_boosting_classifier_with_common_feature(self, jsons):
        feature_mat = self.json_to_feature_vec(jsons)
        pre_classify_results_dic = dict()
        classier_dic = ClassifierConfig.classifier_dic
        for key, value in classier_dic.values():
            classify_result = self.run_classifier_with_common_feature(self.load_model(value), feature_mat)
            pre_classify_results_dic[key] = classify_result
        weight = ClassifierConfig.classifier_weight_dic

        final_classify_result = self.run_boosting(pre_classify_results_dic, weight)
        return final_classify_result

    def run_as_single_classier(self, jsons):
        feature_mat = self.json_to_feature_vec(jsons)
        class_result = self.run_classifier_with_common_feature(
            self.load_model(ClassifierConfig.cur_single_model), feature_mat)
        return class_result

    def run_boosting(self, pre_classify_results_dic, weight):
        pass

    # 将传进来的批量json转换为可用于分类的特征向量矩阵
    def json_to_feature_vec(self, jsons, has_label=False):
        feature_mat = []
        label_vec = []
        # 如果传进来的只是单json，则转换一下
        if type(jsons) is str:
            jsons = [jsons]

        for json in jsons:
            document = Document(json)
            feature_mat.append(self.lexicon.convert_document(document.get_content_words()))
            if has_label:
                label_vec.append(document.label_id)

        if has_label:
            return feature_mat, label_vec
        else:
            return feature_mat

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = ClassifierConfig.svm_model_with_common_feature
        return cPickle.load(open(model_path, 'r'))

    def load_lexicon(self, lexicon_path=None):
        if lexicon_path is None:
            return Lexicon()
        else:
            return cPickle.load(open(lexicon_path, 'r'))

    def load_weight_dic_from_pkl(self):
        return cPickle.load(open(ClassifierConfig.boosting_weight_dic_with_common_feature, 'r'))

    def save_model(self):
        if (ClassifierConfig.is_single_model):
            self.abstract_classifier.save_model()
        else:
            # 保存boosting模型
            pass

    # 从文件加载字典对象
    def load_lexicon_from_pkl(self):
        return cPickle.load(open(FilePathConfig.lexicon_pkl_path, 'r'))

    # 加载类别与编号字典
    # 返回内容类似为{"时政":1,"军事":2,……}
    def load_category_dic_from_pkl(self):
        return cPickle.load(open(FilePathConfig.category_pkl_path, 'r'))

    def save_lexicon_into_pkl(self):
        cPickle.dump(self.lexicon, open(FilePathConfig.lexicon_pkl_path, 'wb'))


if __name__ == '__main__':
    mainClassifier = MainClassifier()
    fid_dic = mainClassifier.construct_lexicon()
    mainClassifier.get_sparse_feature_mat(fid_dic, FilePathConfig.cache_file_path)
