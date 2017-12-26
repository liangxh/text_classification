# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import task2
from task2.model import const
from task2.model.config import config
from nlp.lib.word_embed.build import build_lookup_table, build_vocab_id_mapping


def zero_padding(seq_len):
    def _pad(seq):
        seq += [0] * (seq_len - len(seq))
        return seq
    return _pad


def load_embedding(task_config, return_lookup_table=True):
    # 加載詞嵌入相關數據
    vocab_list = task2.dataset.load_vocab(task_config.task_key, task_config.n_vocab)
    vocab_id_mapping = build_vocab_id_mapping(vocab_list)

    if return_lookup_table:
        if task_config.embedding_algorithm == 'glove':
            from nlp.lib.word_embed.glove import Glove
            embedding = Glove(task_config.embedding_key)
        elif task_config.embedding_algorithm == 'word2vec':
            if task_config.embedding_key == 'frederic_godin':
                from nlp.lib.word_embed.word2vec.frederic_godin import FredericGodinModel
                embedding = FredericGodinModel(config.path_to_frederic_godin_index)
            else:
                raise Exception
        else:
            raise Exception

        task_config.dim_embed = embedding.dim

        # 加載詞嵌入相關模塊
        lookup_table = build_lookup_table(vocab_list, embedding)
        lookup_table = np.asarray(lookup_table)

        return vocab_id_mapping, lookup_table
    else:
        return vocab_id_mapping


def step_train(sess, task_config, nn, dataset):
    """
    進行訓練一輪
    """
    # 準備feed_dict需要的key
    feed_keys = list()
    feed_keys.extend(nn.input_keys())
    feed_keys.append(const.LABEL_GOLD)

    loss = 0.
    label_gold = list()
    label_predict = list()
    for subset_size, subset in dataset.batch_iterate(feed_keys, task_config.batch_size):
        # 準備feed_dict
        feed_dict = dict()
        for key, placeholder in nn.ph_input.items():
            feed_dict[placeholder] = subset[key]
        feed_dict[nn.label_gold] = subset[const.LABEL_GOLD]
        feed_dict[nn.dropout_keep_prob] = task_config.dropout_keep_prob

        # 訓練
        _, partial_loss, partial_labels = sess.run(
            [nn.optimizer, nn.loss, nn.label_predict],
            feed_dict=feed_dict
        )

        # 更新本輪記錄
        loss += partial_loss * subset_size
        label_gold.extend(subset[const.LABEL_GOLD])
        label_predict.extend(partial_labels)

    current_step = tf.train.global_step(sess, nn.global_step)
    loss /= dataset.n_sample
    accuracy = np.mean(np.asarray(label_gold) == np.asarray(label_predict))
    return accuracy, loss, current_step


def step_trial(sess, task_config, nn, dataset):
    """
    進行一輪驗證
    """
    # 準備feed_dict需要的key
    feed_keys = nn.input_keys()
    feed_keys.append(const.LABEL_GOLD)

    loss = 0.
    label_gold = list()
    label_predict = list()

    for subset_size, subset in dataset.batch_iterate(feed_keys, task_config.batch_size):
        # 準備feed_dict
        feed_dict = dict()
        feed_dict[nn.dropout_keep_prob] = 1.
        for key, placeholder in nn.ph_input.items():
            feed_dict[placeholder] = subset[key]
        feed_dict[nn.label_gold] = subset[const.LABEL_GOLD]

        # 驗證
        partial_loss, partial_labels = sess.run(
            [nn.loss, nn.label_predict],
            feed_dict=feed_dict
        )

        # 更新本輪記錄
        loss += partial_loss * subset_size
        label_gold.extend(subset[const.LABEL_GOLD])
        label_predict.extend(partial_labels)

    accuracy = np.mean(np.asarray(label_gold) == np.asarray(label_predict))
    loss /= dataset.n_sample
    return accuracy, loss


def step_test(sess, task_config, nn, dataset):
    feed_keys = nn.input_keys()
    label_predict = list()

    for subset_size, subset in dataset.batch_iterate(feed_keys, task_config.batch_size, shuffle=False):
        # 準備feed_dict
        feed_dict = dict()
        feed_dict[nn.dropout_keep_prob] = 1.
        for key, placeholder in nn.ph_input.items():
            feed_dict[placeholder] = subset[key]

        # 預測
        partial_labels = sess.run(
            nn.label_predict,
            feed_dict=feed_dict
        )
        label_predict.extend(partial_labels)
    return label_predict
