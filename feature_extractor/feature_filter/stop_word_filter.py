import codecs

from abstractFilter import AbstractFilter
from config.config import FilePathConfig


class StopWordFilter(AbstractFilter):
    priority = 2

    def __init__(self):
        stop_words_file = FilePathConfig.stop_words_path
        self.stop_words = set()

        for line in codecs.open(stop_words_file, 'r', 'utf-8', 'ignore'):
            self.stop_words.add(line.strip())

    def filter(self, raw_words):
        raw_words = [word.strip() for word in raw_words if
                     word not in self.stop_words and len(word.strip()) > 0]
        return raw_words
