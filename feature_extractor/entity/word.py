# coding=UTF-8  
class Word(object):
    def __init__(self, word_id, name):
        self.word_id = word_id
        self.name = name
        self.tf = 0
        self.df = 0
