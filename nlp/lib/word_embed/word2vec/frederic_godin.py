# -*- coding: utf-8-*-
import numpy as np
from nlp.model.config import config


class FredericGodinModel(object):
    @classmethod
    def build_index(
            cls,
            filename_model=config.path_to_word2vec_frederic_godin,
            filename_index=config.path_to_word2vec_frederic_godin_index
            ):
        offset = 0
        target_obj = open(filename_model, 'r')

        with open(filename_model, 'r') as file_obj, open(filename_index, 'w') as out_obj:
            header = file_obj.readline().decode('utf8')
            offset += len(header)

            vocab_size, dim = map(int, header.split())  # throws for invalid file format
            binary_len = np.dtype(np.float32).itemsize * dim
            for line_no in xrange(vocab_size):
                word = []
                while True:
                    ch = file_obj.read(1)
                    offset += 1

                    if ch == b' ':
                        break
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have newline, some don't)
                        word.append(ch)
                word = unicode(b''.join(word), encoding='latin-1')
                offset = file_obj.tell()
                target_obj.read(binary_len)

                out_obj.write('{} {}\n'.format(word, offset).encode('utf8'))


if __name__ == '__main__':
    FredericGodinModel.build_index()
