# coding=UTF-8
import codecs
import sys

sys.path.append("../")
from feature_extractor.entity.document import Document

reload(sys)
sys.setdefaultencoding('UTF-8')

data_path = "../file/"

match_result = codecs.open(data_path + "filter_type_news_with_content.txt", 'r', 'utf-8', 'ignore')

filter_list = ["生活", "公益", "摄影", "职场", "文化", "动漫", "风水", "亲子", "移民", "收藏"]

file_list = []
for type_name in filter_list:
    name = data_path + "type_" + str(filter_list.index(type_name)) + ".txt"
    file_list.append(codecs.open(name, 'w', 'utf-8', 'ignore'))

count = 0
count2 = 0
for line in match_result:
    # print count
    count += 1
    document = Document(line)
    title = document.title
    label = document.label
    words = document.get_filtered_content_words_feature()
    # print label, len(words), title
    if label == None:
        continue
    index = filter_list.index(label)

    if len(words) > 2:
        if label == "风水":
            if ("风水" not in title) and ("风水" not in document.splitContent):
                continue

        file_list[index].write(line)
    else:
        count2 += 1
        print line

print count2
for file_item in file_list:
    file_item.close()
