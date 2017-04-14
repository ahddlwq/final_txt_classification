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

common_filter = CommonFilter()
stop_words_filter = StopWordFilter()
speech_filter = SpeechFilter()

match_result = codecs.open(data_path + "filter_type_data5-15.txt", 'r', 'utf-8', 'ignore')

filter_list = ["生活", "公益", "摄影", "职场", "文化", "动漫", "风水", "亲子", "移民", "收藏"]

file_list = []
for type_name in filter_list:
    name = data_path + "type_" + str(filter_list.index(type_name)) + ".txt"
    file_list.append(codecs.open(name, 'w', 'utf-8', 'ignore'))

count = 0
count2 = 0
for line in match_result:
    print count
    count += 1
    document = Document(line)
    document.add_filter(common_filter).add_filter(stop_words_filter).add_filter(speech_filter)
    title = document.title
    label = document.label
    keywords = document.get_filtered_content_words_feature()
    raw_content = document.get_raw_content()
    # print label, len(words), title
    if label == None or keywords == None:
        continue
    index = filter_list.index(label)

    content = ""
    for keyword in keywords:
        content = content + keyword + ","
    # 去除最后一个逗号
    content = content[:-1]

    # print line
    if len(keywords) > 2:
        if label == "风水":
            if ("风水" not in title) and ("风水" not in document.splitContent):
                continue

        file_list[index].write(document.json + '\t' + content + '\t' + raw_content + '\t' + document.label + '\n')
    else:
        count2 += 1

print count2
for file_item in file_list:
    file_item.close()
