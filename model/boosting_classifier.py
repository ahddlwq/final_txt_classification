# coding=UTF-8
import cPickle

from abstract_classifier import AbstractClassifier
from config.config import ClassifierConfig


class BoostingClassifier(AbstractClassifier):
    def __init__(self):
        self.model_path = ClassifierConfig.boosting_model_path
        self.sub_models = {}

    def classify_top_k(self, document, top_k):
        if self.model is None:
            self.load_model()

        # 先需要获取子分类器的结果
        pre_classify_results_dic = dict()
        classier_dic = ClassifierConfig.classifier_path_dic
        for key, value in classier_dic.values():
            if key not in self.sub_models:
                self.sub_models[key] = self.load_sub_model(value)
            pre_classify_results_dic[key] = self.sub_models[key].predict_top_k(document, top_k)

        final_classify_result = self.run_boosting(pre_classify_results_dic)
        return final_classify_result

    # 跑boosting
    def run_boosting(self, pre_classify_results_dic):
        weight_dic = ClassifierConfig.classifier_weight_dic
        final_classify_result = None
        return final_classify_result

    def load_weight_dic_from_pkl(self):
        return self.util.load_object_from_pkl(ClassifierConfig.boosting_weight_dic)

    def load_sub_model(self, model_path):
        return cPickle.load(model_path)
