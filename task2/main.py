# -*- coding: utf-8 -*-
from __future__ import print_function

import datetime
import importlib
import os
import shutil
import commandr
import tensorflow as tf
import task2
from task2.lib import step
from task2.lib.common import load_embedding
from task2.model.task_config import TaskConfig


def get_algorithm(name):
    module = importlib.import_module('task2.nn.{}'.format(name))
    return module.Algorithm


def check_checkpoint_directory(dir_name):
    if os.path.exists(dir_name):
        print (
            '<INFO> Checkout point directory already exists\n' +
            '\t[0] exit\n' +
            '\t[1] remove it: rm -r {}\n'.format(dir_name) +
            '\t[2] or rename it: mv {}{{,.{}}}'.format(
                dir_name, datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            )
        )
        ret = input('choose: ')
        if ret == 0:
            exit()
        elif ret == 1:
            shutil.rmtree(dir_name)
        elif ret == 2:
            shutil.move(dir_name, dir_name + datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
        else:
            raise Exception('invalid reply: {}'.format(ret))

    os.mkdir(dir_name)


@commandr.command('train')
def train(config_filename):
    # 加載配置
    task_config = TaskConfig.load(config_filename)
    check_checkpoint_directory(task_config.dir_checkpoint)
    shutil.copy(config_filename, os.path.join(task_config.dir_checkpoint, 'config'))

    # 選擇算法
    algorithm = get_algorithm(task_config.algorithm)(task_config)

    # 加載數據
    vocab_id_mapping, lookup_table = load_embedding(task_config)
    dataset_train = algorithm.load_and_prepare_dataset('train', vocab_id_map=vocab_id_mapping.map)
    dataset_trial = algorithm.load_and_prepare_dataset('trial', vocab_id_map=vocab_id_mapping.map)

    # 生成神經網絡
    nn = algorithm.build_neural_network(lookup_table)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        best_dev_accuracy = 0.
        best_epoch = -1

        for epoch in range(task_config.epochs):
            print('[epoch {}]'.format(epoch))

            train_accuracy, train_loss, current_step = step.train(sess, task_config, nn, dataset_train)
            print('TRAIN: loss:{}, acc:{}'.format(train_loss, train_accuracy))

            if (epoch + 1) % task_config.validate_interval == 0:
                trial_accuracy, trial_loss = step.trial(sess, task_config, nn, dataset_trial)
                print('TRIAL: loss:{}, acc:{}'.format(trial_loss, trial_accuracy))

                if trial_accuracy > best_dev_accuracy:
                    best_dev_accuracy = trial_accuracy
                    best_epoch = epoch
                    path = saver.save(sess, task_config.prefix_checkpoint, global_step=current_step)
                    print('new checkpoint saved to {}'.format(path))
    print('')
    print('best_accuracy on dev:{}'.format(best_dev_accuracy))
    print('best_epoch: {}'.format(best_epoch))


@commandr.command('test')
def test(config_filename):
    # 加載配置
    task_config = TaskConfig.load(config_filename)

    # 選擇算法
    algorithm = get_algorithm(task_config.algorithm)(task_config)

    # 加載數據
    vocab_id_mapping = load_embedding(task_config, return_lookup_table=False)
    dataset = algorithm.load_and_prepare_dataset('trial', output=False, vocab_id_map=vocab_id_mapping.map)

    with tf.Session() as sess:
        # 加載模型
        prefix_checkpoint = tf.train.latest_checkpoint(task_config.dir_checkpoint)
        saver = tf.train.import_meta_graph("{}.meta".format(prefix_checkpoint))
        saver.restore(sess, prefix_checkpoint)

        # 摘出測試需要的placeholder
        graph = tf.get_default_graph()
        nn = algorithm.build_from_graph(graph)

        # 預測
        label_predict = step.test(sess, task_config, nn, dataset)

    import numpy as np
    labels = task2.dataset.load_labels(task_config.task_key, 'trial')
    print(np.mean(np.asarray(labels) == np.asarray(label_predict)))


if __name__ == '__main__':
    commandr.Run()
