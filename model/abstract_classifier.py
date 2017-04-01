import cPickle

from config.config import ClassifierConfig
from misc.util import Util
class AbstractClassifier(object):

    def __init__(self):
        self.model = None
        self.model_path = None
        pass

    def classify(self, document):
        return self.classify_top_k(document, 1)

    def classify_top_k(self, documents, top_k):
        if self.model is None:
            self.load_model()

        raw_results = self.model.predict(documents)
        return raw_results

    def save_model(self):
        cPickle.dump(self.model, open(self.model_path, 'w'))

    def load_model(self):
        if not Util.is_file(self.model_path):
            Util.log_tool.log.error("model not exist")
            Util.quit()
        self.model = cPickle.load(open(self.model_path, 'r'))

    def train(self, feature_mat, label_vec):
        self.model = ClassifierConfig.classifier_init_dic[ClassifierConfig.cur_single_model]
        Util.log_tool.log.debug("model training")
        self.model.fit(feature_mat, label_vec)
        self.save_model()
        if hasattr(self.model, 'best_params_'):
            Util.log_tool.log.info(self.model.best_params_)
            Util.log_tool.log.info(self.model.best_score_)
