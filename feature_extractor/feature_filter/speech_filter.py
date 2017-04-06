from abstractFilter import AbstractFilter


class SpeechFilter(AbstractFilter):
    priority = 1

    def __init__(self):
        self.speech_tag_filter = ['i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 'eng', 'x', 'v']
        pass

    def filter(self, raw_words_with_type):
        raw_words_with_type = [word.split('_')[0] for word in raw_words_with_type if
                               len(word.split('_')) == 2 and word.split('_')[1] in self.speech_tag_filter]
        return raw_words_with_type
