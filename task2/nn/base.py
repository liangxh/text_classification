# -*- coding: utf-8 -*-
import tensorflow as tf
from task2.common import zero_padding
from task2.dataset import source_key_to_func
from task2.model.dataset import Dataset
from task2.model import const

feed_keys_input = [const.TOKEN_ID_SEQ, const.SEQ_LEN, const.LEXICON_FEAT]


def load_and_prepare_dataset(task_config, mode, output=True, vocab_id_map=None):
    source_keys = [const.TOKEN_ID_SEQ, const.LEXICON_FEAT]
    if output:
        source_keys.append(const.LABEL_GOLD)

    # 讀取原始數據
    source_dict = dict(map(
        lambda key: (key, source_key_to_func[key](task_config.task_key, mode)),
        source_keys
    ))

    # 數據處理
    dataset = Dataset(source_dict)
    dataset.map(const.TOKEN_ID_SEQ, vocab_id_map)
    dataset.map(const.TOKEN_ID_SEQ, len, const.SEQ_LEN)
    dataset.map(const.TOKEN_ID_SEQ, zero_padding(task_config.seq_len))

    # 填补config
    task_config.dim_lexicon_feat = dataset.get_dim(const.LEXICON_FEAT)
    if output:
        task_config.dim_output = dataset.get_dim(const.LABEL_GOLD)

    return dataset


def build_common_part(model_config, loss):
    # 權重調整
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        model_config.learning_rate_initial,
        global_step=global_step,
        decay_steps=model_config.learning_rate_decay_steps,
        decay_rate=model_config.learning_rate_decay_rate
    )
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    return global_step, optimizer


def get_graph_elements_for_test(tf_graph, feed_keys_input):
    keys = [
        const.DROPOUT_KEEP_PROB,
        const.LABEL_PREDICT
    ]
    keys.extend(feed_keys_input)

    elements = dict()
    for key in keys:
        elements[key] = tf_graph.get_operation_by_name(key).outputs[0]
    return elements
