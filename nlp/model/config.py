# -*- coding: utf-8 -*-
import os


class Config(object):
    dir_lab = os.path.join(os.environ['HOME'], 'lab')
    dir_glove = os.path.join(dir_lab, 'glove')

    def path_to_glove(self, key):
        return os.path.join(self.dir_glove, '{}.txt'.format(key))

    def path_to_glove_index(self, key):
        return os.path.join(self.dir_glove, '{}.index'.format(key))

    path_to_word2vec_frederic_godin = os.path.join(dir_lab, 'word2vec', 'frederic_godin.model')


config = Config()
