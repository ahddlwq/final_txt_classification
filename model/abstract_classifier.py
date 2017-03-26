from config.config import ClassifierConfig
from model.single_classify_result import SingleClassifyResult
class AbstractClassifier(object):
    config = ClassifierConfig()
    def __init__(self):
        self.model = None
        pass

    def classify(self, document):
        return SingleClassifyResult()
        pass

    def classify_top_k(self, document, top_k):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

    def train(self):
        pass
