# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import datetime
import commandr
import tensorflow as tf
import task2
from task2.model.task_config import TaskConfig
from task2.common import load_embedding
from task2.common import step_train, step_trial, step_test
from task2.nn.base import get_graph_elements_for_test


def get_algorithm(name):
    if name == 'gru':
        from task2.nn import gru
        return gru
    else:
        raise Exception('invalid algorithm name: {}'.format(name))


def check_checkpoint_directory(dir_name):
    if os.path.exists(dir_name):
        raise Exception(
            'Checkout point directory already exists\n' +
            '\tremove it: rm -r {}\n'.format(dir_name) +
            '\tor rename it: mv {}/{{,.{}}}'.format(
                dir_name, datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            )
        )
    else:
        os.mkdir(dir_name)


@commandr.command('train')
def train(config_filename):
    # 加載配置
    task_config = TaskConfig.load(config_filename)
    check_checkpoint_directory(task_config.dir_checkpoint)

    # 選擇算法
    algorithm = get_algorithm(task_config.algorithm)

    # 加載數據
    vocab_id_mapping, lookup_table = load_embedding(task_config)
    dataset_train = algorithm.load_and_prepare_dataset(task_config, 'train', vocab_id_map=vocab_id_mapping.map)
    dataset_trial = algorithm.load_and_prepare_dataset(task_config, 'trial', vocab_id_map=vocab_id_mapping.map)

    # 生成神經網絡
    nn = algorithm.build_neural_network(task_config, lookup_table)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        best_dev_accuracy = 0.

        for epoch in range(task_config.epochs):
            print('epoch: {}\t'.format(epoch))

            train_accuracy, train_loss, current_step = step_train(sess, task_config, nn, dataset_train, algorithm.feed_keys_input)
            print('TRAIN: loss:{}, acc:{}'.format(train_loss, train_accuracy))

            if (epoch + 1) % task_config.validate_interval == 0:
                trial_accuracy, trial_loss = step_trial(sess, task_config, nn, dataset_trial, algorithm.feed_keys_input)
                print('TRIAL: loss:{}, acc:{}'.format(trial_loss, trial_accuracy))

                if trial_accuracy > best_dev_accuracy:
                    best_dev_accuracy = trial_accuracy
                    path = saver.save(sess, task_config.prefix_checkpoint, global_step=current_step)
                    print('new checkpoint saved to {}'.format(path))


@commandr.command('test')
def test(config_filename):
    # 加載配置
    task_config = TaskConfig.load(config_filename)

    # 選擇算法
    algorithm = get_algorithm(task_config.algorithm)

    # 加載數據
    vocab_id_mapping = load_embedding(task_config, return_lookup_table=False)
    dataset = algorithm.load_and_prepare_dataset(task_config, 'trial', output=False, vocab_id_map=vocab_id_mapping.map)

    with tf.Session() as sess:
        # 加載模型
        prefix_checkpoint = tf.train.latest_checkpoint(task_config.dir_checkpoint)
        print(prefix_checkpoint)
        saver = tf.train.import_meta_graph("{}.meta".format(prefix_checkpoint))
        saver.restore(sess, prefix_checkpoint)

        # 摘出測試需要的placeholder
        graph = tf.get_default_graph()
        nn = get_graph_elements_for_test(graph, algorithm.feed_keys_input)

        # 預測
        label_predict = step_test(sess, task_config, nn, dataset, algorithm.feed_keys_input)

    import numpy as np
    labels = task2.dataset.load_labels(task_config.task_key, 'trial')
    print(np.mean(np.asarray(labels) == np.asarray(label_predict)))


if __name__ == '__main__':
    commandr.Run()
