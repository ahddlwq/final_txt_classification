# coding=UTF-8
import codecs
import sys

sys.path.append("../")
reload(sys)
sys.setdefaultencoding('UTF-8')
from feature_extractor.entity.document import Document

gongyi_data = codecs.open("../file/type_6.txt", 'r', 'utf-8', 'ignore')
filtered_gongyi_data = codecs.open("../file/type_6_filtered.txt", 'w', 'utf-8', 'ignore')

count = 0
for line in gongyi_data:
    label = 0
    document = Document(line)
    title = document.title
    words = document.get_filtered_content_words_feature()

    if len(words) > 6:
        count += 1
        filtered_gongyi_data.write(line)

filtered_gongyi_data.close()
gongyi_data.close()
print count
