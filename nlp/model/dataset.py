# -*- coding: utf-8 -*-
from nlp.lib import index_batch


class Dataset(object):
    def __init__(self, *source_list):
        source_len_list = map(len, source_list)
        for source_len in source_len_list[1:]:
            if not source_len == source_len_list[0]:
                raise Exception('length mismatch: {} != {}'.format(source_len, source_len_list[0]))

        self.source_list = source_list
        self.n_sample = source_len_list[0]

    def batch_iterate(self, batch_size, shuffle=True):
        for indices in index_batch.generate(self.n_sample, batch_size, shuffle):
            subset_list = list()
            for source in self.source_list:
                subset = map(lambda idx: source[idx], indices)
                subset_list.append(subset)
            yield tuple(subset_list)
