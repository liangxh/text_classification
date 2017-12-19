# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import copy
import numpy as np
from lib.dataset.iterate import DatasetIterator
from model.config import config
from optparse import OptionParser
from lib.dataset import task2
from lib.preprocess import vocab
from lib.word_embed.glove import Glove
from lib.word_embed.build import build_lookup_table, build_vocab_id_mapping
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, BasicLSTMCell
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn


class TaskConfig(object):
    glove_key = 'twitter'
    dim_embed = 50
    n_vocab = 8000
    rnn_dim = 100
    learning_rate = 0.01
    epochs = 5
    batch_size = 64
    output_dim = None
    seq_len = None


def build_neural_network(lookup_table):
    model_config = TaskConfig()

    ph_label_gold = tf.placeholder(tf.int32, [None, ])
    ph_vocab_id_seq = tf.placeholder(tf.int32, [None, model_config.seq_len])
    ph_seq_len = tf.placeholder(tf.int32, [None, ])
    ph_dropout_keep_prob = tf.placeholder(tf.float32)
    l2_loss = tf.constant(0.)

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

    label = tf.cast(tf.argmax(y, 1), tf.int32)

    prob_gold = tf.one_hot(ph_label_gold, model_config.output_dim)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=prob_gold))
    optimizer = tf.train.AdamOptimizer(learning_rate=model_config.learning_rate).minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(ph_label_gold, label), tf.float32))

    return ph_vocab_id_seq, ph_seq_len, ph_label_gold, ph_dropout_keep_prob, label, optimizer, accuracy, loss


def run(
        task_key
        ):
    task_config = TaskConfig()

    # 加載訓練數據
    tokenized = task2.load_tokenized(task_key, 'train')
    labels = task2.load_labels(task_key, 'train')

    max_seq_len = max(*map(len, tokenized)) + 1
    output_dim = max(*labels) + 1

    TaskConfig.seq_len = max_seq_len
    TaskConfig.output_dim = output_dim

    vocab_list = vocab.load(task_key, task_config.n_vocab)
    embedding = Glove(task_config.glove_key, task_config.dim_embed)

    vocab_id_mapping = build_vocab_id_mapping(vocab_list)
    lookup_table = build_lookup_table(vocab_list, embedding)

    token_id_seq = map(vocab_id_mapping.map, tokenized)

    ph_vocab_id_seq, ph_seq_len, ph_label_gold, ph_dropout_keep_prob, \
        label, optimizer, accuracy, loss = \
        build_neural_network(lookup_table)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_iterator = DatasetIterator(token_id_seq, labels)
        for epoch in range(task_config.epochs):
            loss_train = 0.
            accuracy_train = 0.
            label_list = list()
            gold_label_list = list()

            print('epoch: {}\t'.format(epoch), end='')
            for input_list, output_list in train_iterator.batch_iterate(task_config.batch_size):
                seq_len = map(len, input_list)

                # 在長度不足的輸入末尾补零
                input_batch = copy.deepcopy(input_list)
                for input_seq in input_batch:
                    input_seq += [0] * (max_seq_len - len(input_seq))

                _, partial_loss, partial_accuracy, partial_label = sess.run(
                    [optimizer, loss, accuracy, label],
                    feed_dict={
                        ph_vocab_id_seq: input_batch,
                        ph_label_gold: output_list,
                        ph_seq_len: seq_len,
                        ph_dropout_keep_prob: 0.5
                    }
                )

                n_sample = len(input_list)
                label_list.extend(partial_label)
                gold_label_list.extend(output_list)

                accuracy_train += partial_accuracy * n_sample
                loss_train += partial_loss * n_sample

            accuracy_train /= train_iterator.n_sample
            loss_train /= train_iterator.n_sample
            print('loss:{}, acc:{}'.format(loss_train, accuracy_train))


def main():
    optparser = OptionParser()
    optparser.add_option('-k', '--key', dest='task_key')
    opts, args = optparser.parse_args()

    run(opts.task_key)


if __name__ == '__main__':
    main()

