# coding=UTF-8
class Term(object):
    def __init__(self, term_id, weight):
        self.term_id = term_id
        self.weight = weight

    def __cmp__(self, other):
        return cmp(self.weight, other.weight)
