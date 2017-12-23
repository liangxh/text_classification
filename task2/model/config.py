# -*- coding: utf-8 -*-
import os


class Config(object):
    dir_lab = os.path.join(os.environ['HOME'], 'lab')
    dir_task2 = os.path.join(dir_lab, 'task2')
    dir_train_checkpoint = os.path.join(dir_task2, 'model_checkpoint')

    path_to_frederic_godin_index = os.path.join(dir_task2, 'word2vec', 'frederic_godin.index')


config = Config()
