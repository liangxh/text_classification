# -*- coding: utf-8 -*-


class NNPack(object):
    def __init__(self,
                 ph_input,
                 dropout_keep_prob=None,
                 label_gold=None, label_predict=None,
                 loss=None, optimizer=None, global_step=None, **kwargs
                 ):
        self.ph_input = ph_input
        self.dropout_keep_prob = dropout_keep_prob
        self.label_gold = label_gold
        self.label_predict = label_predict
        self.loss = loss
        self.global_step = global_step
        self.optimizer = optimizer
        self.other = kwargs

    def get(self, key):
        return self.other[key]

    def input_keys(self):
        return self.ph_input.keys()

    def get_input(self, key):
        return self.ph_input[key]
