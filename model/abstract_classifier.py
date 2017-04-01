# coding=UTF-8
import cPickle

import numpy as np

from config.config import ClassifierConfig
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
        self.model.fit(feature_mat, label_vec)
        self.save_model()
        if hasattr(self.model, 'best_params_'):
            Util.log_tool.log.info(self.model.best_params_)
            Util.log_tool.log.info(self.model.best_score_)
