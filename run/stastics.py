# # coding=UTF-8
# import codecs
# import sys
#
# import json
#
# reload(sys)
# sys.setdefaultencoding('UTF-8')
#
# title_label_dic = {}
#
# data = codecs.open("../file/" + "top10000.txt", 'r', 'utf-8', 'ignore')
#
# sta_dic = {}
# count = 0
# for line in data:
#     json_object = json.loads(line)
#     if json_object == None:
#         continue
#     count += 1
#     if count % 1000 == 0:
#         print count
#     if json_object.has_key("features"):
#         list2 = json_object["features"]
#         index = 0
#         while index + 2 < len(list2):
#             if list2[index + 1] == 'c':
#                 item = list2[index]
#                 if item not in sta_dic:
#                     sta_dic[item] = 0
#                 sta_dic[item] += 1
#             else:
#                 break
#             index += 3
#
# data.close()
#
# for key, value in sta_dic.items():
#     print key, value
# coding=UTF-8
import codecs
import json
import sys

reload(sys)
sys.setdefaultencoding('UTF-8')

title_label_dic = {}

data = codecs.open("../file/top10000.txt", 'r', 'utf-8', 'ignore')

filter_data = codecs.open("../file/filter_type_data.txt", 'w', 'utf-8', 'ignore')

sta_dic = {}
filter_list = ["生活", "公益", "摄影", "职场", "文化", "动漫", "风水", "亲子", "移民", "收藏"]
count = 0
for line in data:
    json_object = json.loads(line)
    if json_object is None:
        continue
    count += 1
    if count % 1000 == 0:
        print count
    if json_object.has_key("category"):
        list2 = json_object["category"]
        for item in list2:
            if item not in sta_dic:
                sta_dic[item] = 0

            if item in filter_list:
                filter_data.write(line.strip() + "\t" + item + "\n")
            sta_dic[item] += 1

data.close()
filter_data.close()

for key, value in sta_dic.items():
    print key, value
