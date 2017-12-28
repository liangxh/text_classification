# -*- coding: utf-8 -*-
"""
訓練/測試過程中用於遍歷數據和獲取數據相關信息
"""
import random


def generate_index_batch(n, batch_size, shuffle=True, round_end=False):
    indices = range(n)
    if shuffle:
        random.shuffle(indices)

    idx = 0
    while idx < n:
        next_idx = idx + batch_size
        ret_indices = indices[idx:next_idx]

        if round_end and len(ret_indices) < batch_size:
            size_diff = batch_size - len(ret_indices)
            patch_indices = indices[:idx]
            random.shuffle(patch_indices)
            ret_indices.extend(patch_indices[:size_diff])

        yield ret_indices
        idx = next_idx


class Dataset(object):
    def __init__(self, source_dict):
        source_len_list = map(len, source_dict.values())
        for source_len in source_len_list[1:]:
            if not source_len == source_len_list[0]:
                raise Exception('length mismatch: {} != {}'.format(source_len, source_len_list[0]))

        self.sources = source_dict
        self.n_sample = source_len_list[0]

    def batch_num(self, batch_size):
        return len(list(generate_index_batch(self.n_sample, batch_size)))

    def batch_iterate(self, keys, batch_size, shuffle, round_end):
        """
        [注] 由于部分神經網絡要求每一輪的batch_size相同, round_end=True在最后一輪當數據不足batch_size時隨機補滿
        """
        for indices in generate_index_batch(self.n_sample, batch_size, shuffle, round_end):
            subset_dict = dict()
            for key in keys:
                source = self.sources[key]
                subset = map(lambda idx: source[idx], indices)
                subset_dict[key] = subset
            yield len(indices), subset_dict

    def get_dim(self, source_key):
        source = self.sources[source_key]
        if isinstance(source[0], int):
            return max(*source) + 1
        else:
            return len(source[0])

    def map(self, source_key, func, output_key=None):
        if output_key is None:
            output_key = source_key

        data = self.sources[source_key]
        data = map(func, data)
        self.sources[output_key] = data

    def get(self, source_key):
        return self.sources[source_key]


if __name__ == '__main__':
    for b in generate_index_batch(10, 3, shuffle=True, round_end=True):
        print b
