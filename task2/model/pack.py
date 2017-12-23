# -*- coding: utf-8 -*-


class NeuralNetworkPack(object):
    def __init__(self,
                 token_id_seq=None, lexicon_feat=None, seq_len=None, label_gold=None, dropout_keep_prob=None,
                 count_correct=None, loss=None, label_predict=None,
                 global_step=None, optimizer=None
                 ):
        # input of the network
        self.token_id_seq = token_id_seq
        self.lexicon_feat = lexicon_feat
        self.seq_len = seq_len
        self.label_gold = label_gold
        self.dropout_keep_prob = dropout_keep_prob

        # output of the network
        self.count_correct = count_correct
        self.loss = loss
        self.label_predict = label_predict

        # other
        self.global_step = global_step
        self.optimizer = optimizer
