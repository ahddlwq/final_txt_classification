# coding=UTF-8
import json
import re

from config.config import ClassifierConfig
from feature_extractor.word_extractor.bigram_extractor import BiGramExtractor
from feature_extractor.word_extractor.common_word_extractor import CommonWordExtractor


class Document(object):
    def __init__(self, raw_document):
        split_data = raw_document.split('\t')
        json_data = split_data[0]

        json_object = json.loads(json_data)
        self.json = json_data
        self.splitContent = json_object['splitContent']
        self.id = json_object['ID']

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

        self.words = None
        self.raw_content = None
        self.label = None

        if len(split_data) > 1:
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
    def get_content_words_feature(self):
        # if len(self.words) == 0:
        #     raw_content = self.abstract_extractor.extract(self.raw_content)
        # else:
        #     raw_content = self.words
        content = self.splitContent
        content = re.sub('_[A-Za-z]+', '', content)
        words = content.split()
        words = [word.lower() for word in words if len(word.strip()) > 1]
        return words

    def get_new_feature(self):
        pass

    # 从正文中取出词，并过滤
    def get_filtered_content_words_feature(self):
        content = self.splitContent
        # content = re.sub('_[A-Za-z]+', '', content)
        content = re.sub('http://(.*).jpg', '', content)
        content = re.sub('http://(.*).gif', '', content)
        content = re.sub('http://(.*).undefined', '', content)
        content = re.sub('http://(.*)search', '', content)
        content = re.sub('http://(.*)tml', '', content)
        content = re.sub('http://(.*)html', '', content)
        content = re.sub('http://(.*)shtml', '', content)
        content = re.sub('http://(.*).jpeg', '', content)
        content = re.sub('http://(.*)com', '', content)
        content = re.sub('http://(.*)cn', '', content)
        content = re.sub('http://(.*).png', '', content)
        content = re.sub('http://(.*)net', '', content)
        content = re.sub('_(.*).jpg', '', content)
        content = re.sub('_(.*).jpeg', '', content)
        content = re.sub('_(.*).png', '', content)
        content = re.sub('_(.*).gif', '', content)
        content = re.sub('_(.*).undefined', '', content)
        content = re.sub('_(.*)html', '', content)
        content = re.sub('_(.*).tml', '', content)
        content = re.sub('_(.*)shtml', '', content)
        content = re.sub('_(.*).com', '', content)
        content = re.sub('_(.*).cn', '', content)
        content = re.sub('_(.*).net', '', content)
        content = re.sub('deg', '', content)
        content = re.sub('nbsp', '', content)
        content = re.sub('quot', '', content)
        content = re.sub('middot', '', content)
        content = re.sub('&', '', content)
        content = re.sub('http://[0-9A-Za-z?=\-\.\/]+', '', content)
        content = re.sub('http://(.*)/', '', content)
        content = content.replace('\n', '')
        # content = content.replace(' ', '')

        content = content.split()
        # 对添加的filter进行排序，使优先级高的先进行过滤
        sorted(self.filters)
        for filter in self.filters:
            content = filter.filter(content)

        return content
