# -*- coding: utf-8 -*-
import tensorflow as tf
from task2.model import const
from task2.model.dataset import Dataset
from task2.common import zero_padding
from task2.nn.base import BaseAlgorithm
from task2.dataset import source_key_to_func


class Algorithm(BaseAlgorithm):
    feed_keys_input = [const.TOKEN_ID_SEQ, const.SEQ_LEN]

    def load_and_prepare_dataset(self, mode, output=True, vocab_id_map=None):
        source_keys = [const.TOKEN_ID_SEQ, ]
        if output:
            source_keys.append(const.LABEL_GOLD)

        # 讀取原始數據
        source_dict = dict(map(
            lambda key: (key, source_key_to_func[key](self.config.task_key, mode)),
            source_keys
        ))

        # 數據處理
        dataset = Dataset(source_dict)
        dataset.map(const.TOKEN_ID_SEQ, vocab_id_map)
        dataset.map(const.TOKEN_ID_SEQ, len, const.SEQ_LEN)
        dataset.map(const.TOKEN_ID_SEQ, zero_padding(self.config.seq_len))

        # 填补config
        if output:
            self.config.dim_output = dataset.get_dim(const.LABEL_GOLD)

        return dataset

    def build_neural_network(self, lookup_table):
        token_id_seq = tf.placeholder(tf.int32, [None, self.config.seq_len], name=const.TOKEN_ID_SEQ)
        seq_len = tf.placeholder(tf.int32, [None, ], name=const.SEQ_LEN)
        dropout_keep_prob = tf.placeholder(tf.float32, name=const.DROPOUT_KEEP_PROB)
        lookup_table = tf.Variable(lookup_table, dtype=tf.float32, name=const.LOOKUP_TABLE)

        embedded = tf.nn.embedding_lookup(lookup_table, token_id_seq)
        rnn_outputs, rnn_last_states = tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.GRUCell(self.config.dim_rnn),
            inputs=embedded,
            sequence_length=seq_len,
            dtype=tf.float32
        )

        rnn_output = tf.nn.dropout(rnn_last_states, dropout_keep_prob)
        dense_input = rnn_output
        w = tf.Variable(tf.truncated_normal(
            [dense_input.shape[-1].value, self.config.dim_output], stddev=0.1)
        )
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

