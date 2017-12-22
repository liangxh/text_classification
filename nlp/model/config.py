# -*- coding: utf-8 -*-
import os


class Config(object):
    dir_lab = os.path.join(os.environ['HOME'], 'lab')
    dir_glove = os.path.join(dir_lab, 'glove')

    def path_to_glove(self, key, dim):
        return os.path.join(self.dir_glove, '{}.{}d.txt'.format(key, dim))

    def path_to_glove_index(self, key, dim):
        return os.path.join(self.dir_glove, '{}.{}d.index'.format(key, dim))

    path_to_word2vec_frederic_godin = os.path.join(dir_lab, 'word2vec', 'frederic_godin.model')
    path_to_word2vec_frederic_godin_index = os.path.join(dir_lab, 'word2vec', 'frederic_godin.index')


config = Config()
