# coding=UTF-8
import codecs
import sys

sys.path.append("../")
from feature_extractor.entity.document import Document
from feature_extractor.feature_filter.common_filter import CommonFilter
from feature_extractor.feature_filter.speech_filter import SpeechFilter
from feature_extractor.feature_filter.stop_word_filter import StopWordFilter

reload(sys)
sys.setdefaultencoding('UTF-8')

data_path = "../file/"

data = codecs.open(data_path + "match_result.txt", 'r', 'utf-8', 'ignore')

match_result = codecs.open(data_path + "new_corpus.txt", 'w', 'utf-8', 'ignore')

common_filter = CommonFilter()
stop_words_filter = StopWordFilter()
speech_filter = SpeechFilter()
count = 0
for line in data:
    print count
    count += 1
    document = Document(line)
    document.add_filter(common_filter).add_filter(stop_words_filter).add_filter(speech_filter)
    keywords = document.get_filtered_content_words_feature()
    if keywords is None:
        continue
    raw_content = document.get_raw_content()
    content = ""
    for keyword in keywords:
        content = content + keyword + ","
    # 去除最后一个逗号
    content = content[:-1]
    # 提取出内容切词和原文，重新写入文件
    match_result.write(document.json + '\t' + content + '\t' + raw_content + '\t' + document.label + '\n')

match_result.close()
