class AbstractFilter(object):
    priority = -1

    def __init__(self):
        pass

    def filter(self, content_words):
        pass

    def __cmp__(self, other):
        return self.priority < other.priority
