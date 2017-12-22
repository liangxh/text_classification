# -*- coding: utf-8-*-
import numpy as np
import json
from nlp.model.config import config


class FredericGodinModel(object):
    def __init__(self, index):
        self.file_model = open(config.path_to_word2vec_frederic_godin, 'r')

        header = self.file_model.readline()
        vocab_size, dim = map(int, header.split())
        self.dim = dim
        self.index = index
        self._binary_len = np.dtype(np.float32).itemsize * dim

    def get(self, vocab):
        if isinstance(vocab, str):
            vocab = vocab.decode('utf8')
        offset = self.index.get(vocab)
        if offset is None:
            return None
        else:
            self.file_model.seek(offset)
            bytes_vec = self.file_model.read(self._binary_len)
            vec = np.fromstring(bytes_vec, dtype=np.float32)
            return vec

    @classmethod
    def load_by_specific_index(cls, filename):
        index = json.load(open(filename, 'r'))
        return cls(index)

    @classmethod
    def build_specific_index(
            cls, vocab_list,
            filename_model=config.path_to_word2vec_frederic_godin,
            ):
        vocab_set = set(vocab_list)
        index = dict()

        with open(filename_model, 'r') as file_obj:
            header = file_obj.readline().decode('utf8')
            vocab_size, dim = map(int, header.split())  # throws for invalid file format
            binary_len = np.dtype(np.float32).itemsize * dim
            for line_no in xrange(vocab_size):
                ch_list = []
                while True:
                    ch = file_obj.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have newline, some don't)
                        ch_list.append(ch)
                token = unicode(b''.join(ch_list), encoding='latin-1')
                offset = file_obj.tell()

                if token in vocab_set:
                    index[token] = offset
                    if len(index) == len(vocab_set):
                        break
                file_obj.read(binary_len)

        return index

    @classmethod
    def test_build_specific_index(
            cls, filename_model=config.path_to_word2vec_frederic_godin,
            ):

        with open(filename_model, 'r') as file_obj, open(filename_model, 'r') as mirror_obj:
            header = file_obj.readline().decode('utf8')
            vocab_size, dim = map(int, header.split())  # throws for invalid file format
            binary_len = np.dtype(np.float32).itemsize * dim
            for line_no in xrange(vocab_size):
                ch_list = []
                while True:
                    ch = file_obj.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have newline, some don't)
                        ch_list.append(ch)
                token = unicode(b''.join(ch_list), encoding='latin-1')
                offset = file_obj.tell()
                bytes_vec = file_obj.read(binary_len)

                mirror_obj.seek(offset)
                mirror_bytes_vec = mirror_obj.read(binary_len)

                if not bytes_vec == mirror_bytes_vec:
                    raise Exception('Test failed')

                if line_no >= 100:
                    print '100 test passed'
                    return


if __name__ == '__main__':
    FredericGodinModel.test_build_specific_index()
