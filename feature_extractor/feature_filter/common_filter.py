from abstractFilter import AbstractFilter


class CommonFilter(AbstractFilter):
    priority = 0

    def __init__(self, is_lower=True):
        self.is_lower = is_lower
        pass

    def filter(self, raw_words):
        raw_words_with_type = [word for word in raw_words if len(word.strip()) > 1]
        if self.is_lower:
            raw_words = [word.lower() for word in raw_words_with_type]
        return raw_words
