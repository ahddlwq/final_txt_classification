from abstractFilter import AbstractFilter


class CommonFilter(AbstractFilter):
    priority = 0

    def __init__(self, is_lower=True):
        self.is_lower = is_lower
        pass

    def filter(self, raw_words_with_type):
        if self.is_lower:
            raw_words = [word.lower() for word in raw_words_with_type if len(word.strip()) > 1]
        else:
            raw_words = [word for word in raw_words_with_type if len(word.strip()) > 1]
        return raw_words
