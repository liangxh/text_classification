# -*- coding: utf-8 -*-
from .exceptions import DifferentVectorSizeException
from .token_id import TokenIdMapping
import numpy as np


class WordEmbedding(object):
    def __init__(self, dim, token_list, vec_list):
        self.dim = dim
        self.token_list = token_list
        self.vec_list = vec_list

    def get_token_id_mapping(self):
        return TokenIdMapping(self.token_list)

    @staticmethod
    def load(filename, generate_unknown=True):
        dim = None
        token_list = list()
        vec_list = list()
        with open(filename, 'r') as fobj:
            for line in fobj:
                line = line.strip()
                if line == '':
                    continue
                parts = line.split(' ')
                token = parts[0]
                vec = parts[1:]
                if dim is None:
                    dim = len(vec)
                elif not len(vec) == dim:
                    raise DifferentVectorSizeException('{} != {}: {}'.format(dim, len(vec), line))
                token_list.append(token)
                vec_list.append(vec)

        if generate_unknown:
            new_token_list = [TokenIdMapping.TOKEN_UNKNOWN, ]
            new_token_list.extend(token_list)
            token_list = new_token_list

            new_vec_list = [np.random.random(dim), ]
            new_vec_list.extend(vec_list)
            vec_list = new_vec_list

        return WordEmbedding(dim, token_list, vec_list)
