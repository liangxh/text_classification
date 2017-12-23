# -*- coding: utf-8 -*-
import copy
import numpy as np
import task2
from task2.model.config import config
from nlp.lib.word_embed.build import build_lookup_table, build_vocab_id_mapping


def input_list_to_batch(input_list, seq_len):
    # 在長度不足的輸入末尾补零
    input_batch = copy.deepcopy(input_list)
    for input_seq in input_batch:
        input_seq += [0] * (seq_len - len(input_seq))
    return input_batch


def load_embedding(key, task_config):
    # 加載詞嵌入相關數據
    vocab_list = task2.dataset.load_vocab(key, task_config.n_vocab)

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
    vocab_id_mapping = build_vocab_id_mapping(vocab_list)
    lookup_table = build_lookup_table(vocab_list, embedding)
    lookup_table = np.asarray(lookup_table)
    return vocab_id_mapping, lookup_table


def step_train(sess, task_config, nn, dataset):
    """
    進行訓練一輪

    sess: tensorflow中的Session實例
    nn: NeuralNetworkPack實例
    """

    loss = 0.
    count_correct = 0.
    for token_id_seq, lexicon_feat, labels in dataset.batch_iterate(task_config.batch_size):
        seq_len = map(len, token_id_seq)
        token_id_batch = input_list_to_batch(token_id_seq, task_config.seq_len)

        _, partial_loss, partial_count_correct = sess.run(
            [nn.optimizer, nn.loss, nn.count_correct],
            feed_dict={
                nn.token_id_seq: token_id_batch,
                nn.lexicon_feat: lexicon_feat,
                nn.label_gold: labels,
                nn.seq_len: seq_len,
                nn.dropout_keep_prob: 1.
            }
        )
        n_sample = len(labels)
        count_correct += partial_count_correct
        loss += partial_loss * n_sample

    accuracy = count_correct / dataset.n_sample
    loss /= dataset.n_sample
    return accuracy, loss


def step_trial(sess, task_config, nn, dataset):
    """
    進行一輪驗證

    sess: tensorflow中的Session實例
    nn: NeuralNetworkPack實例
    """
    loss = 0.
    count_correct = 0.
    for token_id_seq, lexicon_feat, labels in dataset.batch_iterate(task_config.batch_size, shuffle=False):
        seq_len = map(len, token_id_seq)
        token_id_batch = input_list_to_batch(token_id_seq, task_config.seq_len)

        partial_loss, partial_count_correct = sess.run(
            [nn.loss, nn.count_correct],
            feed_dict={
                nn.token_id_seq: token_id_batch,
                nn.lexicon_feat: lexicon_feat,
                nn.label_gold: labels,
                nn.seq_len: seq_len,
                nn.dropout_keep_prob: 1.
            }
        )
        n_sample = len(labels)
        count_correct += partial_count_correct
        loss += partial_loss * n_sample

    accuracy = count_correct / dataset.n_sample
    loss /= dataset.n_sample
    return accuracy, loss
