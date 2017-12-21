# -*- coding: utf-8 -*-
import os


class Config(object):
    path_to_root = os.path.join(os.path.dirname(__file__), '..')
    dir_lab = os.path.join(os.environ['HOME'], 'lab')
    dir_glove = os.path.join(dir_lab, 'glove')
    dir_task2 = os.path.join(dir_lab, 'task2')

    def path_to_glove(self, key, dim):
        return os.path.join(self.dir_glove, '{}.{}d.txt'.format(key, dim))

    def path_to_glove_index(self, key, dim):
        return os.path.join(self.dir_glove, '{}.{}d.index'.format(key, dim))

    def path_to_vocab(self, key):
        return os.path.join(self.dir_task2, 'vocab/{}'.format(key))


config = Config()
