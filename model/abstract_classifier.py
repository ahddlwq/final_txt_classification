# coding=UTF-8
import cPickle

import numpy as np

from config.config import ClassifierConfig, FilePathConfig
from misc.util import Util

class AbstractClassifier(object):

    def __init__(self):
        self.model = None
        self.model_path = None
        pass

    def classify_top_k(self, documents, top_k):
        if self.model is None:
            self.load_model()
        # 检查top_k
        if top_k < 1:
            top_k = 1
        top_k = int(top_k)

        final_results = []
        # 如果当前分类器支持返回概率，则走概率
        if (top_k == 1) | (ClassifierConfig.cur_single_model not in ClassifierConfig.can_predict_pro_classifiers):
            raw_results = self.model.predict(documents)
            for raw_result in raw_results:
                final_results.append([(raw_result, 1)])
        else:
            pro_of_lines = self.model.predict_proba(documents)
            for pro_of_line in pro_of_lines:
                line_result = []
                # 降序获得概率最大的
                pro_index = np.argsort(-pro_of_line)
                for i in range(top_k):
                    line_result.append((self.model.classes_[pro_index[i]], pro_of_line[pro_index[i]]))
                final_results.append(line_result)
        return final_results

    # def classify_top_k(self, documents, top_k):
    #     if self.model is None:
    #         self.load_model()
    #
    #     raw_results = self.model.predict(documents)
    #     return raw_results

    def save_model(self):
        cPickle.dump(self.model, open(self.model_path, 'w'))

    def load_model(self):
        if not Util.is_file(self.model_path):
            Util.log_tool.log.error("model not exist")
            Util.quit()
        else:
            Util.log_tool.log.debug("loading model")
            self.model = cPickle.load(open(self.model_path, 'r'))

    def train(self, feature_mat, label_vec):
        self.model = ClassifierConfig.classifier_init_dic[ClassifierConfig.cur_single_model]
        Util.log_tool.log.debug("model training")
        if ClassifierConfig.cur_single_model == ClassifierConfig.gnb_name:
            self.partial_train(feature_mat, label_vec)
        else:
            self.model.fit(feature_mat, label_vec)

        self.save_model()
        if hasattr(self.model, 'best_params_'):
            Util.log_tool.log.info(self.model.best_params_)
            Util.log_tool.log.info(self.model.best_score_)

    def partial_train(self, feature_mat, label_vec):
        minibatch_train_iterators = self.iter_minibatches(feature_mat, label_vec, minibatch_size=2000)

        for i, (X_train, y_train) in enumerate(minibatch_train_iterators):
            self.model.partial_fit(X_train, y_train)

    def iter_minibatches(self, feature_mat, label_vec, minibatch_size=1000):
        '''
        迭代器
        每次输出minibatch_size行，默认选择1k行
        将输出转化成numpy输出，返回X, y
        '''
        X = []
        y = []
        cur_line_num = 0
        index = 0
        for line in feature_mat:
            x = np.zeros(ClassifierConfig.max_num_features)

            y.append(label_vec[index])

            X.append(x)  # 这里要将数据转化成float类型

            cur_line_num += 1
            index += 1
            if cur_line_num >= minibatch_size:
                X = np.array(X)  # 将数据转成numpy的array类型并返回
                y = np.array(y)
                yield X, y
                X = []
                y = []
                cur_line_num = 0


if __name__ == '__main__':
    abstract = AbstractClassifier()
    feature_mat, label_vec = Util.get_libsvm_data(FilePathConfig.test_feature_mat_path)
    minibatch_train_iterators = abstract.iter_minibatches(feature_mat, label_vec, minibatch_size=2000)
    # for i, (X_train, y_train) in enumerate(minibatch_train_iterators):
