# -*- coding: utf-8 -*-
import os
import time
import json
from model.config import config


class Glove(object):

    def __init__(self, key, dim):
        self.dim = dim
        self.index = self.load_index(key, dim)
        self.file_glove = open(config.path_to_glove(key, dim), 'r')

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
    def dump_index(cls, index, key, dim):
        json.dump(index, open(config.path_to_glove_index(key, dim), 'w'))

    @classmethod
    def load_index(cls, key, dim):
        path = config.path_to_glove_index(key, dim)
        if os.path.exists(path):
            st = time.time()
            index = json.load(open(path, 'r'))
            print 'load index', time.time() - st
        else:
            st = time.time()
            index = cls.build_index(key, dim)
            print 'build index:', time.time() - st

            st = time.time()
            cls.dump_index(index, key, dim)
            print 'write index:', time.time() - st
        return index

    @classmethod
    def build_index(cls, key, dim):
        path = config.path_to_glove(key, dim)
        index = dict()
        with open(path, 'r') as file_obj:
            offset = 0
            for line in file_obj:
                vocab = line[:line.find(' ')]
                index[vocab] = offset
                offset += len(line)
        return index


def test():
    glove = Glove('twitter', 25)
    print glove.get('haha')


if __name__ == '__main__':
    test()
