from abstract_classifier import AbstractClassifier
from config.config import ClassifierConfig

class GBDTClassifier(AbstractClassifier):
    def __init__(self):
        super(GBDTClassifier, self).__init__()
        self.model_path = ClassifierConfig.xgb_model_path
