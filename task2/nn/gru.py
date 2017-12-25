# -*- coding: utf-8 -*-
import tensorflow as tf
from task2.model import const
from task2.nn.base import build_common_part
from task2.nn.base import feed_keys_input, load_and_prepare_dataset  # 外部加載用，勿刪


def build_neural_network(model_config, lookup_table):
    label_gold = tf.placeholder(tf.int32, [None, ], name=const.LABEL_GOLD)
    token_id_seq = tf.placeholder(tf.int32, [None, model_config.seq_len], name=const.TOKEN_ID_SEQ)
    lexicon_feat = tf.placeholder(tf.float32, [None, model_config.dim_lexicon_feat], name=const.LEXICON_FEAT)
    seq_len = tf.placeholder(tf.int32, [None, ], name=const.SEQ_LEN)
    dropout_keep_prob = tf.placeholder(tf.float32, name=const.DROPOUT_KEEP_PROB)
    lookup_table = tf.Variable(lookup_table, dtype=tf.float32, name=const.LOOKUP_TABLE)

    embedded = tf.nn.embedding_lookup(lookup_table, token_id_seq)
    rnn_outputs, rnn_last_states = tf.nn.dynamic_rnn(
        tf.nn.rnn_cell.GRUCell(model_config.dim_rnn),
        inputs=embedded,
        sequence_length=seq_len,
        dtype=tf.float32
    )

    rnn_output = tf.nn.dropout(rnn_last_states, dropout_keep_prob)

    dense_input = tf.concat([rnn_output, lexicon_feat], axis=1)

    w = tf.Variable(tf.truncated_normal(
        [model_config.dim_rnn + model_config.dim_lexicon_feat, model_config.dim_output], stddev=0.1)
    )
    b = tf.Variable(tf.constant(0.1, shape=[model_config.dim_output]))
    y = tf.matmul(dense_input, w) + b

    # 預測標籤
    label_predict = tf.cast(tf.argmax(y, 1), tf.int32, name=const.LABEL_PREDICT)

    # 計算準確率
    count_correct = tf.reduce_sum(tf.cast(tf.equal(label_gold, label_predict), tf.float32), name=const.COUNT_CORRECT)

    # 計算loss
    prob_gold = tf.one_hot(label_gold, model_config.dim_output)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=prob_gold))
    l2_loss = tf.constant(0., dtype=tf.float32)
    l2_loss += tf.nn.l2_loss(w)
    loss += model_config.l2_reg_lambda * l2_loss

    global_step, optimizer = build_common_part(model_config, loss)

    return {
        const.TOKEN_ID_SEQ: token_id_seq,
        const.LEXICON_FEAT: lexicon_feat,
        const.SEQ_LEN: seq_len,
        const.LABEL_GOLD: label_gold,
        const.DROPOUT_KEEP_PROB: dropout_keep_prob,
        const.COUNT_CORRECT: count_correct,
        const.LOSS: loss,
        const.LABEL_PREDICT: label_predict,
        const.GLOBAL_STEP: global_step,
        const.OPTIMIZER: optimizer
    }
