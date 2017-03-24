# coding=UTF-8
from word import Word


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
        return self.id_dic.get(word_id, None)

    def add_document(self, doc_words):
        self.term_set.clear()
        for doc_word in doc_words:
            word = self.name_dic.get(doc_word, None)
            if word is None:
                if self.locked:
                    continue
                word = Word(len(self.name_dic), doc_word)
                self.name_dic[word.name] = word
                self.id_dic[word.word_id] = word

            word.tf += 1
            if word.word_id not in self.term_set:
                self.term_set.add(word.word_id)
                word.df += 1

        self.num_docs += 1

    def convert_document(self, doc_words):
        words = list()
        n = 0
        for doc_word in doc_words:
            word = self.name_dic.get(doc_word, None)
            if word is None:
                if self.locked:
                    continue
                word = Word(len(self.name_dic), doc_word)
                word.tf = 1
                word.df = 1
                self.name_dic[word.name] = word
                self.id_dic[word.id] = word
            words.append(word)
            n += 1

        if n < len(words):
            final_words = list()
            for i in range(0, n):
                final_words.append(words[i])
            words = final_words
        return words

    def __len__(self):
        return len(self.id_dic)

    def save_to_file(self, file_path):
        pass

    def build_word(self):
        pass

    def map(self):
        pass
