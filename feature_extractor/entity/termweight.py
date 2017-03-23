# coding=UTF-8
import math


class TermWeight(object):
    def __init__(self, lexicon):
        self.lexicon = lexicon
        pass


class TfIdfWighter(TermWeight):
    def __init__(self, lexicon):
        TermWeight.__init__(lexicon)

    def cal_weight(self, word_id, tf):
        num_docs = self.lexicon.num_docs
        word = self.lexicon.get_word(word_id)
        return math.log10(tf + 1) * (math.log10(float(num_docs) / word.df + 1))


class TfOnlyTermWeighter(TermWeight):
    def __init__(self, lexicon):
        super(TfOnlyTermWeighter, self).__init__(lexicon)

    def cal_weight(self, word_id, tf):
        return tf
