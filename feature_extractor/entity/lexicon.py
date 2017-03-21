# coding=UTF-8
class Lexicon(object):
    def __init__(self):
        self.id_dic = dict()
        self.name_dic = dict()
        self.locked = False
        self.num_docs = 0
        self.term_set = set()

    def load_from_file(self, file):
        pass

    def get_word(self, word_id):
        return self.id_dic[word_id]

    def add_document(self,doc_words):
        pass

    def convert_document(self, doc_words):
        pass

    def __len__(self):
        return len(self.id_dic)

    def save_to_file(self,file_path):
        pass

    def build_word(self):
        pass

    def map(self):
        pass
