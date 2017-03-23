class AbstractClassifier(object):
    def __init__(self):
        pass

    def classify(self, document):
        pass

    def classify(self, document, top_k):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass
