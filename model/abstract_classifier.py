import cPickle

from config.config import ClassifierConfig


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

        classify_results = []
        raw_results = self.model.predict(documents)
        # print raw_results
        # for i in range(raw_results.shape[0]):
        #     line = raw_results[i]
        #     index = np.argmax(line)
        #     classify_results.append(index)
        # print classify_results
        return raw_results

    # def classify_top_k(self, documents, top_k):
    #     if self.model is None:
    #         self.load_model()
    #
    #     classify_results = []
    #     raw_results = self.model.predict_proba(documents)
    #     print raw_results
    #     for i in range(raw_results[0].shape[1]):
    #         classify_results.append(SingleClassifyResult(i, raw_results[0][1]))
    #     sorted(classify_results)
    #     return classify_results[:top_k]

    def save_model(self):
        cPickle.dump(self.model, open(self.model_path, 'w'))

    def load_model(self):
        self.model = cPickle.load(open(self.model_path, 'r'))

    def train(self, feature_mat, label_vec):
        self.model = ClassifierConfig.classifier_init_dic[ClassifierConfig.cur_single_model]
        print "training"
        self.model.fit(feature_mat, label_vec)
        self.save_model()
