# -*- coding: utf-8 -*-
import tensorflow as tf
from task2.model import const
from task2.nn.base import BaseAlgorithm


class Algorithm(BaseAlgorithm):
    def build_neural_network(self, lookup_table):
        token_id_seq = tf.placeholder(tf.int32, [None, self.config.seq_len], name=const.TOKEN_ID_SEQ)
        lexicon_feat = tf.placeholder(tf.float32, [None, self.config.dim_lexicon_feat], name=const.LEXICON_FEAT)
        seq_len = tf.placeholder(tf.int32, [None, ], name=const.SEQ_LEN)
        dropout_keep_prob = tf.placeholder(tf.float32, name=const.DROPOUT_KEEP_PROB)
        lookup_table = tf.Variable(lookup_table, dtype=tf.float32, name=const.LOOKUP_TABLE)

        embedded = tf.nn.embedding_lookup(lookup_table, token_id_seq)

        # cnn
        filter_shape = [self.config.filter_size, self.config.dim_embed, self.config.filter_num]
        w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[self.config.filter_num]))
        conv = tf.nn.conv1d(embedded, w, stride=1, padding='VALID')
        h = tf.nn.relu(conv + b)   # [batch_size, ?, filter_num]

        # max pooling
        last_state = tf.nn.max_pool(tf.expand_dims(h, -2), ksize=[1, self.config.seq_len - self.config.filter_size + 1, 1, 1], strides=[1] * 4, padding='VALID')
        last_state = tf.reshape(last_state, [-1, last_state.shape[-1].value])

        dense_input = tf.concat([last_state, lexicon_feat], axis=1)

        w = tf.Variable(tf.truncated_normal([dense_input.shape[-1].value, self.config.dim_output], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[self.config.dim_output]))
        y = tf.matmul(dense_input, w) + b

        # 預測標籤
        label_predict = tf.cast(tf.argmax(y, 1), tf.int32, name=const.LABEL_PREDICT)

        # 計算loss
        label_gold = tf.placeholder(tf.int32, [None, ], name=const.LABEL_GOLD)
        prob_gold = tf.one_hot(label_gold, self.config.dim_output)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=prob_gold), name=const.LOSS)
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
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name=const.OPTIMIZER)

        return self.build_from_graph(tf.get_default_graph())
