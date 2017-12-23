# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
import tensorflow as tf
import task2
from task2.model.pack import NeuralNetworkPack
from task2.common import load_embedding, step_train, step_trial
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


def build_neural_network(model_config, lookup_table):
    label_gold = tf.placeholder(tf.int32, [None, ])
    token_id_seq = tf.placeholder(tf.int32, [None, model_config.seq_len])
    lexicon_feat = tf.placeholder(tf.float32, [None, model_config.dim_lexicon_feat])
    seq_len = tf.placeholder(tf.int32, [None, ])
    dropout_keep_prob = tf.placeholder(tf.float32)
    embeddings = tf.Variable(lookup_table, dtype=tf.float32)

    embedded = tf.nn.embedding_lookup(embeddings, token_id_seq)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(model_config.dim_rnn, forget_bias=0.1, state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
        lstm_cell, output_keep_prob=dropout_keep_prob)

    init_state = lstm_cell.zero_state(model_config.batch_size, tf.float32)
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
        lstm_cell, lstm_cell, embedded, seq_len, init_state, init_state)

    output_state_fw, output_state_bw = output_states
    dense_input = tf.concat([output_state_fw.h, output_state_bw.h, lexicon_feat], axis=1)

    w = tf.Variable(tf.truncated_normal([dense_input.shape[-1].value, model_config.dim_output], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[model_config.dim_output]))
    y = tf.matmul(dense_input, w) + b

    # 預測標籤
    label_predict = tf.cast(tf.argmax(y, 1), tf.int32)
    # 計算準確率
    count_correct = tf.reduce_sum(tf.cast(tf.equal(label_gold, label_predict), tf.float32))

    # 計算loss
    prob_gold = tf.one_hot(label_gold, model_config.dim_output)
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

    return NeuralNetworkPack(
        token_id_seq, lexicon_feat, seq_len, label_gold, dropout_keep_prob,
        count_correct, loss, label_predict,
        global_step, optimizer
    )


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
    nn = build_neural_network(task_config, lookup_table)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver(tf.global_variables())
        best_dev_accuracy = 0.

        for epoch in range(task_config.epochs):
            print('epoch: {}\t'.format(epoch))

            # train
            train_accuracy, train_loss = step_train(sess, task_config, nn, dataset_train)
            print('TRAIN: loss:{}, acc:{}'.format(train_loss, train_accuracy))

            current_step = tf.train.global_step(sess, nn.global_step)

            if (epoch + 1) % task_config.validate_interval == 0:
                trial_accuracy, trial_loss = step_trial(sess, task_config, nn, dataset_trial)
                print('TRIAL: loss:{}, acc:{}'.format(trial_loss, trial_accuracy))

                if trial_accuracy > best_dev_accuracy:
                    best_dev_accuracy = trial_accuracy
                    # path = saver.save(sess, config.dir_train_checkpoint, global_step=current_step)
                    # print('new checkpoint saved to {}'.format(path))


if __name__ == '__main__':
    commandr.Run()
