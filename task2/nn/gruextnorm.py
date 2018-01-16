# -*- coding: utf-8 -*-
import tensorflow as tf
from task2.model import const
from task2.nn.base import BaseAlgorithm
from task2.nn.common import dense, rnn_cell


class Algorithm(BaseAlgorithm):
    feed_keys_input = [const.TOKEN_ID_SEQ, const.SEQ_LEN, const.LEXICON_FEAT, const.IS_TRAINING]

    def build_neural_network(self, lookup_table):
        token_id_seq = tf.placeholder(tf.int32, [None, self.config.seq_len], name=const.TOKEN_ID_SEQ)
        lexicon_feat = tf.placeholder(tf.float32, [None, self.config.dim_lexicon_feat], name=const.LEXICON_FEAT)
        seq_len = tf.placeholder(tf.int32, [None, ], name=const.SEQ_LEN)
        dropout_keep_prob = tf.placeholder(tf.float32, name=const.DROPOUT_KEEP_PROB)
        ph_training = tf.placeholder(tf.bool, name=const.IS_TRAINING)

        lookup_table = tf.Variable(lookup_table, dtype=tf.float32, name=const.LOOKUP_TABLE,
                                   trainable=self.config.embedding_trainable)

        embedded = tf.nn.embedding_lookup(lookup_table, token_id_seq)

        dense_input = list()
        for idx, dim in enumerate(self.config.dim_rnn):
            with tf.variable_scope('rnn_{}'.format(idx)):
                rnn_outputs, rnn_last_states = tf.nn.dynamic_rnn(
                    rnn_cell.build_gru(dim, dropout_keep_prob=dropout_keep_prob),
                    inputs=embedded,
                    sequence_length=seq_len,
                    dtype=tf.float32
                )
                last_state = rnn_last_states
                last_state = tf.contrib.layers.batch_norm(last_state, center=True, scale=True, is_training=ph_training)
                dense_input.append(last_state)

        last_lex = lexicon_feat
        for dim in self.config.dim_lex_dense:
            last_lex = dense.batch_norm(last_lex, ph_training, dim, activation=tf.nn.tanh)
        dense_input.append(last_lex)

        dense_input = tf.concat(dense_input, axis=1)
        if self.config.output_bias:
            y, w, b = dense.build(dense_input, self.config.dim_output)
        else:
            y, w = dense.build(dense_input, self.config.dim_output, bias=False)

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