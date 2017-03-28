from abstract_classifier import AbstractClassifier
from config.config import ClassifierConfig


class RFClassifier(AbstractClassifier):
    def __init__(self):
        super(RFClassifier, self).__init__()
        self.model_path = ClassifierConfig.rf_model_path
