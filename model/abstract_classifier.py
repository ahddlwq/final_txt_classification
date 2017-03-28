import cPickle

from config.config import ClassifierConfig
from evaluation.single_classify_result import SingleClassifyResult


class AbstractClassifier(object):

    def __init__(self):
        self.model = None
        self.model_path = None
        pass

    def classify(self, document):
        return self.classify_top_k(document, 1)

    def classify_top_k(self, document, top_k):
        if self.model is None:
            self.load_model()

        classify_results = []
        raw_results = self.model.predict_proba(document)
        for i in range(raw_results[0].shape[1]):
            classify_results.append(SingleClassifyResult(i, raw_results[0][1]))
        sorted(classify_results)
        return classify_results[:top_k]

    def save_model(self):
        cPickle.dump(self.model, open(self.model_path, 'r'))

    def load_model(self):
        self.model = cPickle.load(self.model_path)

    def train(self, feature_mat, label_vec):
        self.model = ClassifierConfig.classifier_init_dic[ClassifierConfig.cur_single_model]
        self.model.fit(feature_mat, label_vec)
        self.save_model()
