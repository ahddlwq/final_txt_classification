from abstractFilter import AbstractFilter


class SpeechFilter(AbstractFilter):
    priority = 0

    def __init__(self):
        pass

    def filter(self, raw_content):
        return raw_content
