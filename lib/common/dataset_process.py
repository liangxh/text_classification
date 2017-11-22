# -*- coding: utf-8 -*-
from lib.common import index_batch


class DataSampleBatchGenerator(object):
    def __init__(self, data_sample_list):
        self.data_sample_list = data_sample_list

    def generate(self, batch_size, shuffle=True):
        for indices in index_batch.generate(len(self.data_sample_list), batch_size, shuffle):
            input_list = [self.data_sample_list[index][0] for index in indices]
            label_list = [self.data_sample_list[index][1] for index in indices]
            yield input_list, label_list
