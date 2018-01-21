# -*- coding: utf-8 -*-
from __future__ import print_function
import importlib
import os
import shutil
import commandr
import tensorflow as tf
import task2
from task2.snapshot import Snapshot
from task2.lib import step, evaluate
from task2.lib.common import load_embedding
from task2.model.task_config import TaskConfig


def get_algorithm(name):
    module = importlib.import_module('task2.nn.{}'.format(name))
    return module.Algorithm


@commandr.command('train')
def train(config_filename):
    # 加載配置
    task_config = TaskConfig.load(config_filename)
    os.mkdir(task_config.dir_checkpoint)
    shutil.copy(config_filename, os.path.join(task_config.dir_checkpoint, 'config.yaml'))

    snapshot = Snapshot(config_filename, task_config.time_mark)
    snapshot.create()

    # 選擇算法
    algorithm = get_algorithm(task_config.algorithm)(task_config)

    # 加載數據
    vocab_id_mapping, lookup_table = load_embedding(task_config)
    dataset_train = algorithm.load_and_prepare_dataset('train', vocab_id_map=vocab_id_mapping.map)
    dataset_trial = algorithm.load_and_prepare_dataset('trial', vocab_id_map=vocab_id_mapping.map)

    # 生成神經網絡
    task_config.set_learning_rate_decay_steps(dataset_train.batch_num(task_config.batch_size))
    nn = algorithm.build_neural_network(lookup_table)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        best_dev_score_dict = None
        best_epoch = -1

        for epoch in range(task_config.epochs):
            print('[epoch {}]'.format(epoch))

            labels_gold, labels_predict, loss, current_step = step.train(sess, task_config, nn, dataset_train)
            score_dict = evaluate.score(labels_gold, labels_predict)
            print('TRAIN: loss:{}, precision:{}, score:{}'.format(
                loss, score_dict['precision'], score_dict['macro_f1']))

            if (epoch + 1) % task_config.validate_interval == 0:
                labels_gold, labels_predict, trial_loss = step.trial(sess, task_config, nn, dataset_trial)
                score_dict = evaluate.score(labels_gold, labels_predict)
                print('TRAIN: loss:{}, precision:{}, score:{}'.format(
                    loss, score_dict['precision'], score_dict['macro_f1']))

                if best_dev_score_dict is None or score_dict['macro_f1'] > best_dev_score_dict['macro_f1']:
                    best_dev_score_dict = score_dict
                    best_epoch = epoch
                    path = saver.save(sess, task_config.prefix_checkpoint, global_step=current_step)
                    print('new checkpoint saved to {}'.format(path))
    print('')
    print('best_epoch: {}'.format(best_epoch))

    col_line = ''
    val_line = ''
    for key, value in best_dev_score_dict.items():
        col_line += '[{}]\t'.format(key)
        val_line += '{:4f}\t'.format(value)
    print(col_line)
    print(val_line)
    print('model has been saved at: {}'.format(task_config.dir_checkpoint))
    snapshot.rename_by_score(best_dev_score_dict['macro_f1'])


@commandr.command('trial')
def trial(dir_checkpoint):
    # 加載配置
    config_filename = os.path.join(dir_checkpoint, 'config.yaml')
    task_config = TaskConfig.load(config_filename)

    # 選擇算法
    algorithm = get_algorithm(task_config.algorithm)(task_config)

    # 加載數據
    vocab_id_mapping = load_embedding(task_config, return_lookup_table=False)
    dataset = algorithm.load_and_prepare_dataset('trial', output=False, vocab_id_map=vocab_id_mapping.map)

    with tf.Session() as sess:
        # 加載模型
        prefix_checkpoint = tf.train.latest_checkpoint(dir_checkpoint)
        saver = tf.train.import_meta_graph("{}.meta".format(prefix_checkpoint))
        saver.restore(sess, prefix_checkpoint)

        # 摘出測試需要的placeholder
        graph = tf.get_default_graph()
        nn = algorithm.build_from_graph(graph)

        # 預測
        labels_predict = step.test(sess, task_config, nn, dataset)

    labels_gold = task2.dataset.load_labels(task_config.task_key, 'trial')
    score_dict = evaluate.score(labels_gold, labels_predict)

    col_line = ''
    val_line = ''
    for key, value in score_dict.items():
        col_line += '[{}]\t'.format(key)
        val_line += '{:4f}\t'.format(value)
    print(col_line)
    print(val_line)


@commandr.command('test')
def test(dir_checkpoint, output_filename):
    # 加載配置
    config_filename = os.path.join(dir_checkpoint, 'config.yaml')
    task_config = TaskConfig.load(config_filename)

    # 選擇算法
    algorithm = get_algorithm(task_config.algorithm)(task_config)

    # 加載數據
    vocab_id_mapping = load_embedding(task_config, return_lookup_table=False)
    dataset = algorithm.load_and_prepare_dataset('test', output=False, vocab_id_map=vocab_id_mapping.map)

    with tf.Session() as sess:
        # 加載模型
        prefix_checkpoint = tf.train.latest_checkpoint(dir_checkpoint)
        saver = tf.train.import_meta_graph("{}.meta".format(prefix_checkpoint))
        saver.restore(sess, prefix_checkpoint)

        # 摘出測試需要的placeholder
        graph = tf.get_default_graph()
        nn = algorithm.build_from_graph(graph)

        # 預測
        labels_predict = step.test(sess, task_config, nn, dataset)

    with open(output_filename, 'w') as file_obj:
        for label in labels_predict:
            file_obj.write('{}\n'.format(label))


if __name__ == '__main__':
    commandr.Run()
