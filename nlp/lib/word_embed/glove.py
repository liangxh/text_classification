# -*- coding: utf-8 -*-
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
            print 'load index', time.time() - st
        else:
            st = time.time()
            index = cls.build_index(key)
            print 'build index:', time.time() - st

            st = time.time()
            cls.dump_index(index, key)
            print 'write index:', time.time() - st
        return index

    @classmethod
    def build_index(cls, key):
        path = config.path_to_glove(key)
        index = dict()
        with open(path, 'r') as file_obj:
            offset = 0
            for line in file_obj:
                vocab = line[:line.find(' ')]
                index[vocab] = offset
                offset += len(line)
        return index


def test():
    glove = Glove('twitter.25d')
    print glove.get('haha')


if __name__ == '__main__':
    test()
