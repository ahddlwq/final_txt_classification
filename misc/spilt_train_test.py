import codecs

from config.config import FilePathConfig
from feature_extractor.entity.document import Document
from util.util import Util

total_corpus = codecs.open(FilePathConfig.test_corpus_path, "r", FilePathConfig.file_encodeing, "strict")
cate_dic = Util.load_object_from_pkl(FilePathConfig.category_pkl_path)

cate_num_dic = dict()

for line in total_corpus:
    document = Document(line)
    label = document.label
    if label not in cate_num_dic:
        cate_num_dic[label] = 0
    else:
        cate_num_dic[label] += 1

for key, value in cate_num_dic.items():
    print key, value
