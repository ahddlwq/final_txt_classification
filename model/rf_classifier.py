import cPickle

from abstract_classifier import AbstractClassifier
from config.config import ClassifierConfig


class RFClassifier(AbstractClassifier):
    def __init__(self):
        AbstractClassifier.__init__(self)
        pass

    def load_model(self):
        self.model = cPickle.load(open(ClassifierConfig.rf_model_with_common_feature, 'r'))

    def classify(self, document):
        pass

    def classify_top_k(self, document, top_k):
        pass
