# -*- coding: utf-8 -*-
import numpy as np


class VocabIdMapping(object):
    """
    vocab轉換成id, 若不在字典中則返回0
    """
    def __init__(self, vocab_list):
        self.mapping = dict([(vocab, idx) for idx, vocab in enumerate(vocab_list)])

    def get_idx(self, token):
        return self.mapping.get(token, -1) + 2

    def map(self, token_list):
        return map(self.get_idx, token_list)


def _build_random_vector(dim):
    return np.random.normal(0., 0.1, dim)


def build_lookup_table(vocab_list, embedding):
    """
    embedding為lib.word_embed.glove.Glove
    """
    lookup_table = list()
    lookup_table.append(np.zeros(embedding.dim))  # embedding vector for END
    lookup_table.append(np.zeros(embedding.dim))  # embedding vector for UNKNOWN
    for vocab in vocab_list:
        vec = embedding.get(vocab)
        if vec is None:
            vec = _build_random_vector(embedding.dim)
        lookup_table.append(vec)
    return lookup_table


def build_vocab_id_mapping(vocab_list):
    return VocabIdMapping(vocab_list)
