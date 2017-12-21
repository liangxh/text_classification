# -*- coding: utf-8 -*-
import os


class Config(object):
    dir_lab = os.path.join(os.environ['HOME'], 'lab')
    dir_task2 = os.path.join(dir_lab, 'task2')


config = Config()
