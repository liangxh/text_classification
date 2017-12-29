# -*- coding: utf-8 -*-
import tensorflow as tf
from task2.model import const
from task2.nn.base import BaseAlgorithm
from task2.nn.common import dense, cnn
from task2.lib.common import zero_padding
from task2.lib.dataset import source_key_to_func
from task2.model.dataset import Dataset


class Algorithm(BaseAlgorithm):
    feed_keys_input = [const.TOKEN_ID_SEQ]

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
        dataset.map(const.TOKEN_ID_SEQ, zero_padding(self.config.seq_len))

        # 填补config
        if output:
            self.config.dim_output = dataset.get_dim(const.LABEL_GOLD)

        return dataset

    def build_neural_network(self, lookup_table):
        token_id_seq = tf.placeholder(tf.int32, [None, self.config.seq_len], name=const.TOKEN_ID_SEQ)
        #lexicon_feat = tf.placeholder(tf.float32, [None, self.config.dim_lexicon_feat], name=const.LEXICON_FEAT)
        #seq_len = tf.placeholder(tf.int32, [None, ], name=const.SEQ_LEN)
        dropout_keep_prob = tf.placeholder(tf.float32, name=const.DROPOUT_KEEP_PROB)
        lookup_table = tf.Variable(lookup_table, dtype=tf.float32, name=const.LOOKUP_TABLE,
                                   trainable=self.config.embedding_trainable)
        embedded = tf.nn.embedding_lookup(lookup_table, token_id_seq)

        # cnn
        last_state = embedded
        conv_output_list = list()
        for i, filter_config in enumerate(self.config.filters):
            conv_output = cnn.build(
                            last_state, filter_config['num'], filter_config['ksize'],
                            tf.nn.tanh if i == len(self.config.filters) - 1 else None
                        )
            conv_output = cnn.max_pooling(conv_output)
            conv_output_list.append(conv_output)
        last_state = tf.concat(conv_output_list, axis=-1)

        last_state = tf.nn.dropout(last_state, dropout_keep_prob)
        y, w, b = dense.build(last_state, self.config.dim_output)

        # 預測標籤
        label_predict = tf.cast(tf.argmax(y, 1), tf.int32, name=const.LABEL_PREDICT)

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
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name=const.OPTIMIZER)

        return self.build_from_graph(tf.get_default_graph())