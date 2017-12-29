# -*- coding: utf-8 -*-
"""
Recurrent Convolutional Neural Networks for Text Classification
Siwei Lai, Liheng Xu, Kang Liu, Jun Zhao
"""
import tensorflow as tf
from task2.model import const
from task2.nn.base import BaseAlgorithm
from task2.nn.common import dense, rnn_cell


class Algorithm(BaseAlgorithm):
    def build_neural_network(self, lookup_table):
        token_id_seq = tf.placeholder(tf.int32, [None, self.config.seq_len], name=const.TOKEN_ID_SEQ)
        lexicon_feat = tf.placeholder(tf.float32, [None, self.config.dim_lexicon_feat], name=const.LEXICON_FEAT)
        seq_len = tf.placeholder(tf.int32, [None, ], name=const.SEQ_LEN)
        dropout_keep_prob = tf.placeholder(tf.float32, name=const.DROPOUT_KEEP_PROB)
        lookup_table = tf.Variable(lookup_table, dtype=tf.float32, name=const.LOOKUP_TABLE,
                                   trainable=self.config.embedding_trainable)
        embedded = tf.nn.embedding_lookup(lookup_table, token_id_seq)

        cell_fw = rnn_cell.build_basic(self.config.dim_context)
        cell_bw = rnn_cell.build_basic(self.config.dim_context)
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, embedded, seq_len,
            cell_fw.zero_state(self.config.batch_size, tf.float32),
            cell_bw.zero_state(self.config.batch_size, tf.float32)
        )
        outputs_fw, outputs_bw = outputs
        last_state = tf.concat([outputs_fw, embedded, outputs_bw], axis=-1)
        last_state = tf.layers.dense(last_state, self.config.dim_hidden, tf.nn.tanh)
        last_state = tf.reduce_max(last_state, axis=1)

        if self.config.with_lexicon:
            last_state = tf.concat([last_state, lexicon_feat], axis=1)

        last_state = tf.nn.dropout(last_state, dropout_keep_prob)
        y, w, b = dense.build(last_state, self.config.dim_output)

        # 計算loss
        label_gold = tf.placeholder(tf.int32, [None, ], name=const.LABEL_GOLD)
        prob_gold = tf.one_hot(label_gold, self.config.dim_output)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=prob_gold), name=const.LOSS)
        if self.config.l2_reg_lambda is not None and self.config.l2_reg_lambda > 0:
            l2_loss = tf.constant(0., dtype=tf.float32)
            l2_loss += tf.nn.l2_loss(w)
            loss += self.config.l2_reg_lambda * l2_loss

        global_step = tf.Variable(0, trainable=False, name=const.GLOBAL_STEP)
        learning_rate = tf.train.exponential_decay(
            self.config.learning_rate_initial,
            global_step=global_step,
            decay_steps=self.config.learning_rate_decay_steps,
            decay_rate=self.config.learning_rate_decay_rate
        )

        # 預測標籤
        tf.cast(tf.argmax(y, 1), tf.int32, name=const.LABEL_PREDICT)

        # Optimizer
        tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name=const.OPTIMIZER)

        return self.build_from_graph(tf.get_default_graph())
