# coding=UTF-8
import codecs
import sys

from config.config import FilePathConfig
from feature_extractor.entity.document import Document

reload(sys)
sys.setdefaultencoding('UTF-8')

title_label_dic = {}

data = codecs.open(FilePathConfig.raw_news_path, 'r', 'utf-8', 'ignore')
labels = codecs.open(FilePathConfig.file_root_path + "label.txt", 'r', 'utf-8', 'ignore')
match_result = codecs.open(FilePathConfig.file_root_path + "match_result.txt", 'w', 'utf-8', 'ignore')

for line in labels:
    title = line.split('\t')[0]
    label = line.split('\t')[1].strip()
    title_label_dic[title] = label

count = 0
for line in data:
    document = Document(line)
    count += 1
    print count
    if document.title not in title_label_dic:
        match_result.write(line.strip() + '\n')

match_result.close()
