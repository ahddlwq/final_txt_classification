# coding=UTF-8
from config.config import ClassifierConfig
from feature_extractor.word_extractor.bigram_extractor import BiGramExtractor
from feature_extractor.word_extractor.common_word_extractor import CommonWordExtractor


class Document(object):
    def __init__(self, raw_document):
        split_data = raw_document.split('\t')
        json_data = split_data[0]

        self.words = []
        self.source = ""
        self.keywords = []
        self.summary = ""
        self.title = ""
        self.tag = []
        self.label_id = -1
        self.label = ""
        self.raw_content = ""

        self.words = split_data[1].strip().split(',')
        self.raw_content = split_data[2].strip()
        self.label = split_data[3].strip()

        if (ClassifierConfig.is_use_bigram):
            self.abstract_extractor = BiGramExtractor()
        else:
            self.abstract_extractor = CommonWordExtractor()

    def get_content_words(self):
        if len(self.words) == 0:
            return self.abstract_extractor.extract(self.raw_content)
        else:
            return self.words

    def get_new_feature(self):
        pass
