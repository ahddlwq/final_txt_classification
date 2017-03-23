import cPickle
import codecs

from config.file_path_config import FilePathConfig


class Util(object):
    def __init__(self):
        self.config = FilePathConfig()
        pass

    def save_cate_dic(self):
        cate_file = codecs.open(self.config.category_file_path, "rb", self.config.file_encodeing, "ignore")
        cate_dic = {}
        cate_id = 0
        for line in cate_file:
            cate_dic[line.strip()] = cate_id
            cate_id += 1
        cPickle.dump(cate_dic, open(self.config.category_pkl_path, 'wb'))


if __name__ == '__main__':
    util = Util()
    util.save_cate_dic()
