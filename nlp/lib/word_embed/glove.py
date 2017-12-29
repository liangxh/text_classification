# -*- coding: utf-8 -*-
from __future__ import print_function
import json
import os
import time

from nlp.model.config import config


class Glove(object):

    def __init__(self, key):
        self.index = self.load_index(key)
        self.file_glove = open(config.path_to_glove(key), 'r')

        line = self.file_glove.readline()
        self.dim = len(line.strip().split(' ')) - 1

    def get(self, vocab):
        if isinstance(vocab, str):
            vocab = vocab.decode('utf8')
        offset = self.index.get(vocab)
        if offset is None:
            return None
        else:
            self.file_glove.seek(offset)
            line = self.file_glove.readline()
            line = line.strip()
            parts = line.split(' ')
            # if not vocab == parts[0]:
            #    raise Exception(u'Index crash: looking for "{}" bug "{}" returned'.format(vocab, parts[0]))
            vec = map(float, parts[1:])
            return vec

    @classmethod
    def dump_index(cls, index, key):
        json.dump(index, open(config.path_to_glove_index(key), 'w'))

    @classmethod
    def load_index(cls, key):
        path = config.path_to_glove_index(key)
        if os.path.exists(path):
            st = time.time()
            index = json.load(open(path, 'r'))
            print('load index: {}'.format(time.time() - st))
        else:
            st = time.time()
            index = cls.build_index(key)
            print('build index: {}'.format(time.time() - st))

            st = time.time()
            cls.dump_index(index, key)
            print('write index: {}'.format(time.time() - st))
        return index

    @classmethod
    def build_index(cls, key):
        path = config.path_to_glove(key)
        index = dict()

        with open(path, 'r') as file_obj:
            while True:
                offset = file_obj.tell()
                line = file_obj.readline()
                if line == '':
                    break

                vocab = line[:line.find(' ')]
                index[vocab] = offset
        return index

    @classmethod
    def test_build_index(cls, key):
        """
        測試build_index 是否正確
        """
        path = config.path_to_glove(key)
        with open(path, 'r') as file_obj, open(path, 'r') as test_obj:
            while True:
                offset = file_obj.tell()
                line = file_obj.readline()
                if line == '':
                    break

                test_obj.seek(offset)
                test_line = test_obj.readline()
                assert(line == test_line)

        print('test passed')


def test():
    Glove.test_build_index('twitter.25d')


if __name__ == '__main__':
    test()
