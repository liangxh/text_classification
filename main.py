# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import numpy as np
from lib.common.dataset_process import DataSampleBatchGenerator
from lib.data_service import api
from lib.common.data_sample import DataSample


def get_vocabulary_size(X):
    return max([max(x) for x in X]) + 1  # plus the 0th word


def fit_in_vocabulary(X, voc_size):
    return [[w for w in x if w < voc_size] for x in X]


def attention(inputs, attention_size):
    inputs_shape = inputs.shape
    seq_length = inputs_shape[1].value
    hidden_size = inputs_shape[2].value

    W = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b = tf.Variable(tf.random_normal([1, attention_size], stddev=0.1))
    u = tf.Variable(tf.random_normal([attention_size, 1], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W)) + b
    vu = tf.matmul(v, u)
    exps = tf.reshape(tf.exp(vu), [-1, seq_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, seq_length, 1]), 1)
    return output, alphas


def dense_layer(inputs, output_size):
    w = tf.Variable(tf.truncated_normal([inputs.get_shape()[1].value, output_size], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[output_size]))
    y = tf.nn.xw_plus_b(inputs, w, b)
    return y


class config(object):
    batch_size = 256
    sequence_length = 250
    hidden_dim = 150
    embedding_dim = 100
    attention_size = 50
    learning_rate = 1e-3
    epochs = 3
    keep_prob = 0.8
    delta = 0.5

train = api.data.get_dataset('imdb', 'train')
test = api.data.get_dataset('imdb', 'test')

train = map(lambda xy: DataSample(*xy), train)
test = map(lambda xy: DataSample(*xy), test)


vocabulary_size = get_vocabulary_size(X_train)
X_test = fit_in_vocabulary(X_test, vocabulary_size)

X_train = zero_pad(X_train, config.sequence_length)
X_test = zero_pad(X_test, config.sequence_length)

batch = tf.placeholder(tf.int32, [None, config.sequence_length])
labels = tf.placeholder(tf.float32, [None])
sequence_length = tf.placeholder(tf.int32, [None])
keep_prob = tf.placeholder(tf.float32)

embeddings = tf.Variable(tf.random_uniform(
                [vocabulary_size, config.embedding_dim], -1., 1.), trainable=True)
batch_embedded = tf.nn.embedding_lookup(embeddings, batch)

rnn_outputs, _ = bi_rnn(
                    GRUCell(config.hidden_dim),
                    GRUCell(config.hidden_dim),
                    inputs=batch_embedded,
                    sequence_length=sequence_length,
                    dtype=tf.float32
                )

rnn_outputs = tf.concat(rnn_outputs, 2)
attention_output, alphas = attention(rnn_outputs, config.attention_size)
drop = tf.nn.dropout(attention_output, keep_prob)
y = dense_layer(drop, 1)
y = tf.squeeze(y)  # TODO

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(
                tf.equal(tf.round(tf.sigmoid(y)), labels),   # TODO
                tf.float32))

# Actual lengths of sequences
seq_len_test = np.array([list(x).index(0) + 1 for x in X_test])  # TODO
seq_len_train = np.array([list(x).index(0) + 1 for x in X_train])


def get_seq_len(seq_list):
    return np.array([list(seq).index(0) + 1 for seq in seq_list])


train_generator = DataSampleBatchGenerator(train)
test_generator = DataSampleBatchGenerator(test)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('start')
    for epoch in range(config.epochs):
        loss_train = 0
        loss_test = 0
        accuracy_train = 0.
        accuracy_test = 0.

        print('epoch: {}\t'.format(epoch), end='')

        batch_num = 0
        for input_list, label_list in train_generator.generate(config.batch_size):
            seq_len = get_seq_len(input_list)
            partial_loss, acc, _ = sess.run(
                [loss, accuracy, optimizer],
                feed_dict={
                    batch: input_list,
                    labels: label_list,
                    sequence_length: seq_len,
                    keep_prob: config.keep_prob
                }
            )
            accuracy_train += acc
            loss_train += partial_loss
            batch_num += 1
        accuracy_train /= batch_num
        loss_train /= batch_num

        batch_num = 0
        for input_list, label_list in test_generator.generate(config.batch_size):
            seq_len = get_seq_len(input_list)
            partial_loss, acc = sess.run(
                [loss, accuracy],
                feed_dict={
                    batch: input_list,
                    labels: label_list,
                    sequence_length: seq_len,
                    keep_prob: 1.
                }
            )
            accuracy_test += acc
            loss_test += partial_loss
            batch_num += 1
        accuracy_test /= batch_num
        loss_test /= batch_num

        print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
            loss_train, loss_test, accuracy_train, accuracy_test
        ))
        saver.save(sess, "model")


