# coding=UTF-8

import numpy as np
from sklearn.externals import joblib

from config.config import ClassifierConfig, FilePathConfig
from util.util import Util


class AbstractClassifier(object):

    def __init__(self):
        self.model = None
        self.model_path = None
        pass

    def classify_top_k(self, feature_mat, top_k):
        if self.model is None:
            self.load_model()
        # 检查top_k
        if top_k < 1:
            top_k = 1
        top_k = int(top_k)

        if ClassifierConfig.cur_single_model in ClassifierConfig.can_partial_train_predict_classifiers:
            return self.partial_classify_top_k(feature_mat, top_k)

        final_results = []
        # 如果当前分类器支持返回概率，则走概率
        if (top_k == 1) | (ClassifierConfig.cur_single_model not in ClassifierConfig.can_predict_pro_classifiers):
            raw_results = self.model.predict(feature_mat)
            for raw_result in raw_results:
                final_results.append([(raw_result, 1)])
        else:
            pro_of_lines = self.model.predict_proba(feature_mat)
            for pro_of_line in pro_of_lines:
                line_result = []
                # 降序获得概率最大的
                pro_index = np.argsort(-pro_of_line)
                for i in range(top_k):
                    line_result.append((self.model.classes_[pro_index[i]], pro_of_line[pro_index[i]]))
                final_results.append(line_result)
        return final_results

    def partial_classify_top_k(self, feature_mat, top_k):
        final_results = []
        minibatch_train_iterators = self.iter_minibatches_only_x(feature_mat, minibatch_size=2000)
        for i, X_train in enumerate(minibatch_train_iterators):
            print "iter ", i
            if top_k == 1:
                raw_results = self.model.predict(X_train)
                for raw_result in raw_results:
                    final_results.append([(raw_result, 1)])
            else:
                pro_of_lines = self.model.predict_proba(X_train)
                for pro_of_line in pro_of_lines:
                    line_result = []
                    # 降序获得概率最大的
                    pro_index = np.argsort(-pro_of_line)
                    for i in range(top_k):
                        line_result.append((self.model.classes_[pro_index[i]], pro_of_line[pro_index[i]]))
                    final_results.append(line_result)
        return final_results

    def save_model(self):
        joblib.dump(self.model, self.model_path)
        # cPickle.dump(self.model, open(self.model_path, 'w'))

    def load_model(self):
        if not Util.is_file(self.model_path):
            Util.log_tool.log.error("model not exist")
            Util.quit()
        else:
            Util.log_tool.log.debug("loading model")
            print self.model_path
            self.model = joblib.load(self.model_path)
            # self.model = cPickle.load(open(self.model_path, 'r'))

    def train(self, feature_mat, label_vec):
        self.model = ClassifierConfig.classifier_init_dic[ClassifierConfig.cur_single_model]
        Util.log_tool.log.debug("model training")
        if ClassifierConfig.cur_single_model in ClassifierConfig.can_partial_train_predict_classifiers:
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
            print "iter ", i
            self.model.partial_fit(X_train, y_train, classes=np.array(range(0, 30)))

    def iter_minibatches(self, feature_mat, label_vec, minibatch_size=1000):
        '''
        迭代器
        每次输出minibatch_size行，默认选择1k行
        将输出转化成numpy输出，返回X, y
        '''
        X = []
        y = []
        cur_line_num = 0
        for index in xrange(feature_mat.shape[0]):
            x = feature_mat.getrow(index).toarray()
            X.append(x[0])
            y.append(label_vec[index])

            cur_line_num += 1
            if cur_line_num >= minibatch_size:
                # 将数据转成numpy的array类型并返回
                X = np.array(X)
                y = np.array(y)
                yield X, y
                X = []
                y = []
                cur_line_num = 0

        if len(y) > 0:
            yield X, y

    def iter_minibatches_only_x(self, feature_mat, minibatch_size=2000):
        X = []
        cur_line_num = 0
        for index in xrange(feature_mat.shape[0]):
            x = feature_mat.getrow(index).toarray()
            # 得到的是一个[[]]的矩阵，把里面的第一个[]取出来就是需要的单行特征向量
            X.append(x[0])
            cur_line_num += 1
            if cur_line_num >= minibatch_size:
                X = np.array(X)
                yield X
                X = []
                cur_line_num = 0
        if len(X) > 0:
            yield X


if __name__ == '__main__':
    abstract = AbstractClassifier()
    print "loading"
    feature_mat, label_vec = Util.get_libsvm_data(FilePathConfig.test_feature_mat_path)
    print "iter"
    minibatch_train_iterators = abstract.iter_minibatches(feature_mat, label_vec, minibatch_size=2000)
    for i, (X_train, y_train) in enumerate(minibatch_train_iterators):
        if i == 1:
            print [x for x in X_train[0] if x > 0], y_train
            # print i, X_train.shape, y_train.shape
