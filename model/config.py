# -*- coding: utf-8 -*-
import os


class Config(object):
    path_to_root = os.path.join(os.path.dirname(__file__), '..')
    path_to_task2 = os.path.join(os.environ['HOME'], 'lab/task2')
    path_to_local = os.path.join(path_to_root, 'local')
    path_to_glove_dir = os.path.join(os.environ['HOME'], 'lab/glove')

    def path_to_glove(self, key, dim):
        return os.path.join(self.path_to_glove_dir, '{}.{}d.txt'.format(key, dim))

    def path_to_glove_index(self, key, dim):
        return os.path.join(self.path_to_glove_dir, '{}.{}d.index'.format(key, dim))

    def path_to_vocab(self, key):
        return os.path.join(self.path_to_local, 'task2.{}.vocab'.format(key))

    def path_to_tokenized(self, key):
        return os.path.join(self.path_to_local, 'task2.{}.tokenized'.format(key))

config = Config()