# coding=UTF-8
import codecs
from Queue import PriorityQueue

import numpy as np

from abstract_select_function import AbstractSelectFunction
from config.config import FilePathConfig, ClassifierConfig
from feature_extractor.entity.term import Term


class ChiSquare(AbstractSelectFunction):
    def __init__(self):
        pass

    def feature_select(self, lexicon, num_categories, num_doc):
        print "selection"
        cache_file_path = FilePathConfig.cache_file_path
        data = codecs.open(cache_file_path, 'rb', FilePathConfig.file_encodeing, 'ignore')
        # 默认是int64,但是其实用int32就够了，节省内存，int32最大能够到21亿，我们最大只需要几百万
        feature_stat = np.zeros((len(lexicon.name_dic), num_categories), dtype=np.int32)
        feature_freq = np.zeros(len(lexicon.name_dic), dtype=np.int32)
        class_size = np.zeros(num_categories, dtype=np.int32)

        terms = list()
        num_docs_read = 0
        try:
            for line in data:
                # 清空列表
                del terms[:]
                splited_line = line.strip().split(FilePathConfig.sparse_content_split_label)
                if not len(splited_line) == 2:
                    print "Error cache error"
                label_id = int(splited_line[0])
                # 末尾会有回车符
                id_weight_pairs = splited_line[1].strip().split(FilePathConfig.sparse_content_id_weight_list_label)
                # print len(id_weight_pairs)
                for id_weight_pair in id_weight_pairs:
                    term_id_tf = id_weight_pair.split(FilePathConfig.sparse_content_id_weight_label)
                    term = Term(int(term_id_tf[0]), float(term_id_tf[1]))
                    terms.append(term)

                class_size[label_id] += 1
                for term in terms:
                    feature_stat[term.term_id][label_id] += 1
                    feature_freq[term.term_id] += 1
                if num_docs_read % 500 == 0:
                    print "sanned", num_docs_read
                num_docs_read += 1
        except Exception, e:
            print e.message

        data.close()

        print "start cal chi_square", num_doc, num_categories
        selected_features_queue = PriorityQueue(ClassifierConfig.max_num_features + 1)
        for i in range(0, len(lexicon.name_dic)):
            word = lexicon.get_word(i)
            if word is not None:
                if word.df == 1 | len(word.name) > 50:
                    continue
            chi_sqr = -1
            chi_max = -1
            for j in range(0, num_categories):
                # 由于乘法会导致精度问题，需要转换为int64
                A = int(feature_stat[i][j])
                B = int(feature_freq[i] - A)
                C = int(class_size[j] - A)
                D = int(num_doc - A - B - C)

                fractor_base = (A + C) * (B + D) * (A + B) * (C + D) * 1.0
                if fractor_base == 0:
                    chi_sqr = 0
                else:
                    # 不用num_docs，因为都一样
                    chi_sqr = ((A * D - B * C) * (A * D - B * C)) * 1.0 / fractor_base
                if chi_sqr > chi_max:
                    chi_max = chi_sqr

            term = Term(i, chi_max)
            selected_features_queue.put(term)
            if selected_features_queue.qsize() > ClassifierConfig.max_num_features:
                selected_features_queue.get()
        return selected_features_queue
