# -*- coding: utf-8 -*-
from __future__ import print_function

import copy
import commandr
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import dynamic_rnn as rnn

import task2
from task2.model.config import config
from nlp.lib.word_embed.build import build_lookup_table, build_vocab_id_mapping
from nlp.model.dataset import Dataset


class TaskConfig(object):
    task_key = 'us2'
    embedding_algorithm = 'glove'
    embedding_key = 'twitter.50d'
    n_vocab = 4000
    dim_rnn = 50
    learning_rate_initial = 0.01
    learning_rate_decay_rate = 1.
    learning_rate_decay_steps = 10
    l2_reg_lambda = 0.2
    epochs = 5
    validate_interval = 1
    batch_size = 64
    seq_len = 50
    dim_output = None
    dim_lexicon_feat = None
    dim_embed = None


def build_neural_network(lookup_table):
    model_config = TaskConfig()

    ph_label_gold = tf.placeholder(tf.int32, [None, ])
    ph_token_id_seq = tf.placeholder(tf.int32, [None, model_config.seq_len])
    ph_lexicon_feat = tf.placeholder(tf.float32, [None, model_config.dim_lexicon_feat])
    ph_seq_len = tf.placeholder(tf.int32, [None, ])
    ph_dropout_keep_prob = tf.placeholder(tf.float32)

    embeddings = tf.Variable(np.asarray(lookup_table), trainable=True, dtype=tf.float32)
    embedded = tf.nn.embedding_lookup(embeddings, ph_token_id_seq)
    rnn_outputs, rnn_last_states = rnn(
        GRUCell(model_config.dim_rnn),
        inputs=embedded,
        sequence_length=ph_seq_len,
        dtype=tf.float32
    )

    rnn_output = tf.nn.dropout(rnn_last_states, ph_dropout_keep_prob)

    dense_input = tf.concat([rnn_output, ph_lexicon_feat], axis=1)

    w = tf.Variable(tf.truncated_normal([model_config.dim_rnn + model_config.dim_lexicon_feat, model_config.dim_output], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[model_config.dim_output]))
    y = tf.matmul(dense_input, w) + b

    # 預測標籤
    label = tf.cast(tf.argmax(y, 1), tf.int32)
    # 計算準確率
    count_correct = tf.reduce_sum(tf.cast(tf.equal(ph_label_gold, label), tf.float32))

    # 計算loss
    prob_gold = tf.one_hot(ph_label_gold, model_config.dim_output)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=prob_gold))
    l2_loss = tf.constant(0., dtype=tf.float32)
    l2_loss += tf.nn.l2_loss(w)
    loss += model_config.l2_reg_lambda * l2_loss

    # 權重調整
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
                        model_config.learning_rate_initial,
                        global_step=global_step,
                        decay_steps=model_config.learning_rate_decay_steps,
                        decay_rate=model_config.learning_rate_decay_rate
                    )
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    return ph_token_id_seq, ph_lexicon_feat, ph_seq_len, ph_label_gold, ph_dropout_keep_prob, \
        count_correct, loss, label, \
        global_step, optimizer


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
    return vocab_id_mapping, lookup_table


def load_and_prepare_dataset(key, mode, vocab_id_mapping, output=True):
    # 讀取原始數據
    tokenized = task2.dataset.load_tokenized(key, mode)
    lexicon_feat = task2.dataset.load_lexicon_feature(key, mode)

    # 數據加工
    token_id_seq = map(vocab_id_mapping.map, tokenized)

    # 根據是不需要
    if not output:
        return Dataset(token_id_seq, lexicon_feat)
    else:
        labels = task2.dataset.load_labels(key, mode)
        return Dataset(token_id_seq, lexicon_feat, labels)


@commandr.command('train')
def train():
    task_config = TaskConfig()

    vocab_id_mapping, lookup_table = load_embedding(task_config.task_key, task_config)

    TaskConfig.dim_output = task2.dataset.get_output_dim(task_config.task_key)
    TaskConfig.dim_lexicon_feat = task2.dataset.get_lexicon_feature_dim(task_config.task_key)

    dataset_train = load_and_prepare_dataset(task_config.task_key, 'train', vocab_id_mapping)
    dataset_trial = load_and_prepare_dataset(task_config.task_key, 'trial', vocab_id_mapping)

    # 生成神經網絡
    ph_token_id_seq, ph_lexicon_feat, ph_seq_len, ph_label_gold, ph_dropout_keep_prob, \
        ret_count_correct, ret_loss, ret_label, \
        global_step, optimizer = \
        build_neural_network(lookup_table)

    def step_train(dataset):
        loss = 0.
        count_correct = 0.
        for token_id_seq, lexicon_feat, labels in dataset.batch_iterate(task_config.batch_size):
            seq_len = map(len, token_id_seq)
            token_id_batch = input_list_to_batch(token_id_seq, TaskConfig.seq_len)

            _, partial_loss, partial_count_correct = sess.run(
                [optimizer, ret_loss, ret_count_correct],
                feed_dict={
                    ph_token_id_seq: token_id_batch,
                    ph_lexicon_feat: lexicon_feat,
                    ph_label_gold: labels,
                    ph_seq_len: seq_len,
                    ph_dropout_keep_prob: 1.
                }
            )
            n_sample = len(labels)
            count_correct += partial_count_correct
            loss += partial_loss * n_sample

        accuracy = count_correct / dataset.n_sample
        loss /= dataset.n_sample
        return accuracy, loss

    def step_trial(dataset):
        loss = 0.
        count_correct = 0.
        for token_id_seq, lexicon_feat, labels in dataset.batch_iterate(task_config.batch_size, shuffle=False):
            seq_len = map(len, token_id_seq)
            token_id_batch = input_list_to_batch(token_id_seq, TaskConfig.seq_len)

            partial_loss, partial_count_correct = sess.run(
                [ret_loss, ret_count_correct],
                feed_dict={
                    ph_token_id_seq: token_id_batch,
                    ph_lexicon_feat: lexicon_feat,
                    ph_label_gold: labels,
                    ph_seq_len: seq_len,
                    ph_dropout_keep_prob: 1.
                }
            )
            n_sample = len(labels)
            count_correct += partial_count_correct
            loss += partial_loss * n_sample

        accuracy = count_correct / dataset.n_sample
        loss /= dataset.n_sample
        return accuracy, loss

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver(tf.global_variables())
        best_dev_accuracy = 0.

        for epoch in range(task_config.epochs):
            print('epoch: {}\t'.format(epoch))

            # train
            train_accuracy, train_loss = step_train(dataset_train)
            print('TRAIN: loss:{}, acc:{}'.format(train_loss, train_accuracy))

            current_step = tf.train.global_step(sess, global_step)

            if (epoch + 1) % TaskConfig.validate_interval == 0:
                trial_accuracy, trial_loss = step_trial(dataset_trial)
                print('TRIAL: loss:{}, acc:{}'.format(trial_loss, trial_accuracy))

                if trial_accuracy > best_dev_accuracy:
                    best_dev_accuracy = trial_accuracy
                    # path = saver.save(sess, config.dir_train_checkpoint, global_step=current_step)
                    # print('new checkpoint saved to {}'.format(path))


if __name__ == '__main__':
    commandr.Run()
