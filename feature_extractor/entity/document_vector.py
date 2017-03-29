# coding=UTF-8
import math

from term import Term


class DocumentVector(object):
    def __init__(self, term_weighter):
        self.term_weighter = term_weighter

    def build(self, words, normalized=False):
        terms_dic = dict()
        for word in words:
            term = terms_dic.get(word.word_id, None)
            if term is None:
                term = Term(word.word_id, 0)
                terms_dic[term.term_id] = term
            term.weight += 1

        vec = list()
        i = 0
        normalizer = 0
        for term in terms_dic.values():
            term.weight = self.term_weighter.cal_weight(term.term_id, term.weight)
            vec.append(term)
            normalizer += term.weight * term.weight
            i += 1

        if normalized:
            normalizer = math.sqrt(normalizer)
            for term in vec:
                term.weight /= normalizer

        sorted(vec)
        return vec
