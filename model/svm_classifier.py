from abstract_classifier import AbstractClassifier
from config.config import ClassifierConfig

class SVMClassifier(AbstractClassifier):
    def __init__(self):
        super(SVMClassifier, self).__init__()
        self.model_path = ClassifierConfig.svm_model_path
