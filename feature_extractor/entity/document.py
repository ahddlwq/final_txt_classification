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

        self.filters = []

        # 目前因为已经把数据处理好，节省时间，所以就按这种方式取
        self.words = split_data[1].strip().split(',')
        self.raw_content = split_data[2].strip()
        self.label = split_data[3].strip()

        if (ClassifierConfig.is_use_bigram):
            self.abstract_extractor = BiGramExtractor()
        else:
            self.abstract_extractor = CommonWordExtractor()

    def add_filter(self, added_filter):
        self.filters.append(added_filter)
        return self

    # 目前因为已经把数据处理好，所以加了一层检验
    def get_content_words(self):
        if len(self.words) == 0:
            raw_content = self.abstract_extractor.extract(self.raw_content)
        else:
            raw_content = self.words

        # 对添加的filter进行排序，使优先级高的先进行过滤
        sorted(self.filters)
        for filter in self.filters:
            raw_content = filter.filter(raw_content)
        return raw_content

    def get_new_feature(self):
        pass
