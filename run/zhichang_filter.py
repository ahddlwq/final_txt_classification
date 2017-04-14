# coding=UTF-8
import codecs
import sys

sys.path.append("../")
reload(sys)
sys.setdefaultencoding('UTF-8')
from feature_extractor.entity.document import Document

gongyi_data = codecs.open("../file/type_3.txt", 'r', 'utf-8', 'ignore')
filtered_gongyi_data = codecs.open("../file/type_3_filtered.txt", 'w', 'utf-8', 'ignore')
key_words = ["职场", "工作", "员工", "老板", "上司", "招聘", "加班", "面试", "就业", "求职", "辞职"]
count = 0
for line in gongyi_data:
    label = 0
    document = Document(line)
    title = document.title
    words = document.splitContent
    words = document.get_filtered_content_words_feature()

    if len(words) <= 6:
        continue
    for word in key_words:
        if (word in title) or (word in words):
            label = 1
            break

    if label == 1:
        count += 1
        filtered_gongyi_data.write(line)
    else:
        print line

filtered_gongyi_data.close()
gongyi_data.close()
print count
