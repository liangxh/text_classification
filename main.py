# -*- coding: utf-8 -*-
from __future__ import print_function

import copy
from optparse import OptionParser

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import dynamic_rnn as rnn

from lib.dataset import task2
from lib.preprocess import vocab
from lib.word_embed.build import build_lookup_table, build_vocab_id_mapping
from lib.word_embed.glove import Glove
from model.dataset import Dataset


class TaskConfig(object):
    glove_key = 'twitter'
    dim_embed = 50
    n_vocab = 8000
    rnn_dim = 100
    learning_rate_initial = 0.01
    learning_rate_decay_rate = 1.
    learning_rate_decay_steps = 10
    l2_reg_lambda = 0.2
    epochs = 5
    validate_interval = 1
    batch_size = 64
    output_dim = None
    seq_len = None


def build_neural_network(lookup_table):
    model_config = TaskConfig()

    ph_label_gold = tf.placeholder(tf.int32, [None, ])
    ph_vocab_id_seq = tf.placeholder(tf.int32, [None, model_config.seq_len])
    ph_seq_len = tf.placeholder(tf.int32, [None, ])
    ph_dropout_keep_prob = tf.placeholder(tf.float32)

    embeddings = tf.Variable(np.asarray(lookup_table), trainable=True, dtype=tf.float32)
    embedded = tf.nn.embedding_lookup(embeddings, ph_vocab_id_seq)
    rnn_outputs, rnn_last_states = rnn(
        GRUCell(model_config.rnn_dim),
        inputs=embedded,
        sequence_length=ph_seq_len,
        dtype=tf.float32
    )

    rnn_output = tf.nn.dropout(rnn_last_states, ph_dropout_keep_prob)

    w = tf.Variable(tf.truncated_normal([model_config.rnn_dim, model_config.output_dim], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[model_config.output_dim]))
    y = tf.matmul(rnn_output, w) + b

    # 預測標籤
    label = tf.cast(tf.argmax(y, 1), tf.int32)
    # 計算準確率
    count_correct = tf.reduce_sum(tf.cast(tf.equal(ph_label_gold, label), tf.float32))

    # 計算loss
    prob_gold = tf.one_hot(ph_label_gold, model_config.output_dim)
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

    return ph_vocab_id_seq, ph_seq_len, ph_label_gold, ph_dropout_keep_prob, \
        count_correct, loss, label, \
        optimizer


def input_list_to_batch(input_list, seq_len):
    # 在長度不足的輸入末尾补零
    input_batch = copy.deepcopy(input_list)
    for input_seq in input_batch:
        input_seq += [0] * (seq_len - len(input_seq))
    return input_batch


def run(
        task_key
        ):
    task_config = TaskConfig()

    # 加載訓練數據
    train_tokenized = task2.load_tokenized(task_key, 'train')
    train_labels = task2.load_labels(task_key, 'train')

    trial_tokenized = task2.load_tokenized(task_key, 'trial')
    trial_labels = task2.load_labels(task_key, 'trial')

    # 加載詞嵌入相關模塊
    vocab_list = vocab.load(task_key, task_config.n_vocab)
    embedding = Glove(task_config.glove_key, task_config.dim_embed)

    vocab_id_mapping = build_vocab_id_mapping(vocab_list)
    lookup_table = build_lookup_table(vocab_list, embedding)

    # 填充配置
    TaskConfig.seq_len = max(
        max(*map(len, train_tokenized)),
        max(*map(len, trial_tokenized)),
    ) + 1

    TaskConfig.output_dim = max(*train_labels) + 1

    train_token_id_seq = map(vocab_id_mapping.map, train_tokenized)
    trial_token_id_seq = map(vocab_id_mapping.map, trial_tokenized)

    dataset_train = Dataset(train_token_id_seq, train_labels)
    dataset_trial = Dataset(trial_token_id_seq, trial_labels)

    # 生成神經網絡
    ph_vocab_id_seq, ph_seq_len, ph_label_gold, ph_dropout_keep_prob, \
        ret_count_correct, ret_loss, ret_label, \
        optimizer = \
        build_neural_network(lookup_table)

    def step_train(dataset):
        loss = 0.
        count_correct = 0.
        for input_list, output_list in dataset.batch_iterate(task_config.batch_size):
            seq_len = map(len, input_list)
            input_batch = input_list_to_batch(input_list, TaskConfig.seq_len)

            _, partial_loss, partial_count_correct = sess.run(
                [optimizer, ret_loss, ret_count_correct],
                feed_dict={
                    ph_vocab_id_seq: input_batch,
                    ph_label_gold: output_list,
                    ph_seq_len: seq_len,
                    ph_dropout_keep_prob: 1.
                }
            )
            n_sample = len(input_list)
            count_correct += partial_count_correct
            loss += partial_loss * n_sample

        accuracy = count_correct / dataset.n_sample
        loss /= dataset.n_sample
        return accuracy, loss

    def step_trial(dataset):
        loss = 0.
        count_correct = 0.
        label_list = list()
        for input_list, output_list in dataset.batch_iterate(task_config.batch_size, shuffle=False):
            seq_len = map(len, input_list)
            input_batch = input_list_to_batch(input_list, TaskConfig.seq_len)

            partial_loss, partial_count_correct, partial_label = sess.run(
                [ret_loss, ret_count_correct, ret_label],
                feed_dict={
                    ph_vocab_id_seq: input_batch,
                    ph_label_gold: output_list,
                    ph_seq_len: seq_len,
                    ph_dropout_keep_prob: 1.
                }
            )
            n_sample = len(input_list)
            count_correct += partial_count_correct
            loss += partial_loss * n_sample
            label_list.extend(partial_label)

        accuracy = count_correct / dataset.n_sample
        loss /= dataset.n_sample
        return accuracy, loss, label_list

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(task_config.epochs):
            print('epoch: {}\t'.format(epoch))

            # train
            train_accuracy, train_loss = step_train(dataset_train)
            print('TRAIN: loss:{}, acc:{}'.format(train_loss, train_accuracy))

            if (epoch + 1) % TaskConfig.validate_interval == 0:
                trial_accuracy, trial_loss, label_list = step_trial(dataset_trial)
                print('TRIAL: loss:{}, acc:{}'.format(trial_loss, trial_accuracy))


def main():
    optparser = OptionParser()
    optparser.add_option('-k', '--key', dest='task_key')
    opts, args = optparser.parse_args()

    run(opts.task_key)


if __name__ == '__main__':
    main()

