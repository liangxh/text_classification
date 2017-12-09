# -*- coding: utf-8 -*-


class DataSample(object):
    def __init__(self, seq, label):
        self.seq = seq
        self.label = label

    def zero_pad(self, length):
        """
        使對序列結尾設置0
        """
        return DataSample(self.seq[:length - 1] + [0] * max(length - len(self.seq), 1), label)

    def padded_len(self):
        """
        返回序列長度，默認 0為序列終結符
        """
        return self.seq.index(0) + 1


class DataSampleBatch(object):
    def __init__(self, data_sample_list):
        self.data_sample_list = data_sample_list

    def zero_pad(self):
        #self.padded_sample_list = map(lambda sample: sample.zero_pad(), self.data_sample_list)
        pass
