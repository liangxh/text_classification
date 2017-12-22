# -*- coding: utf-8 -*-
import os


class Config(object):
    dir_lab = os.path.join(os.environ['HOME'], 'lab')
    dir_task2 = os.path.join(dir_lab, 'task2')

    @classmethod
    def path_to_frederic_godin_index(cls, key):
        return os.path.join(cls.dir_task2, 'word2vec', '{}.frederic_godin.index'.format(key))


config = Config()
