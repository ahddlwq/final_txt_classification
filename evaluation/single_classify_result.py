class SingleClassifyResult(object):
    def __init__(self, label_id, probability):
        self.label_id = label_id
        self.probability = probability

    def __str__(self):
        return "label:" + self.label_id + ",probability:" + self.probability

    def __cmp__(self, other):
        return self.probability > other.probality
