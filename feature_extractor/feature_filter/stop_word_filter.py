from abstractFilter import AbstractFilter


class StopWordFilter(AbstractFilter):
    priority = 1

    def __init__(self):
        pass

    def filter(self, raw_content):
        return raw_content
