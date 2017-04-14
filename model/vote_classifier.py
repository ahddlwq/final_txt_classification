# coding=UTF-8
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib

from abstract_classifier import AbstractClassifier
from config.config import ClassifierConfig
from util.util import Util


# 这是用于投票分类的分类器
class VoteClassifier(AbstractClassifier):
    def __init__(self):
        super(VoteClassifier, self).__init__()
        self.model_path = ClassifierConfig.boosting_model_path
        base_model_names = ClassifierConfig.boosting_using_classifiers
        base_models = []
        base_model_weights = []
        for base_model_name in base_model_names:
            model_path = ClassifierConfig.classifier_path_dic[base_model_name]
            if not Util.is_file(model_path):
                # 如果base模型不存在，则跳过
                continue
            Util.log_tool.log.debug("vote add " + base_model_name)

            model = joblib.load(model_path)
            base_models.append((base_model_name, model))
            base_model_weights.append(ClassifierConfig.classifier_weight_dic[base_model_name])

        self.model = VotingClassifier(estimators=base_models, voting='soft', weights=base_model_weights,
                                      n_jobs=ClassifierConfig.cpu_counts)

    def train(self, feature_mat, label_vec):
        Util.log_tool.log.debug("vote model training")
        self.model.fit(feature_mat, label_vec)
        self.save_model()
