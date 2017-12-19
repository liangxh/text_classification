# -*- coding: utf-8 -*-
from lib.common import index_batch


class DatasetIterator(object):
    def __init__(self, input_list, output_list):
        self.input_list = input_list
        self.output_list = output_list
        self.n_sample = len(self.input_list)

        if not len(self.input_list) == len(self.output_list):
            raise Exception('number of input and number of output mismatches {} != {}'.format(
                len(self.input_list), len(self.output_list)
            ))

    def batch_iterate(self, batch_size, shuffle=True):
        for indices in index_batch.generate(self.n_sample, batch_size, shuffle):
            input_list = [self.input_list[index] for index in indices]
            output_list = [self.output_list[index] for index in indices]
            yield input_list, output_list
