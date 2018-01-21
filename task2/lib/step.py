# -*- coding: utf-8 -*-
import tensorflow as tf
from task2.model import const
from progressbar import ProgressBar, Percentage, Bar, ETA


def progressbar(maxval):
    widgets = ['Progress: ', Percentage(), ' ', Bar(marker='>'), ' ', ETA()]
    return ProgressBar(widgets=widgets, maxval=maxval).start()


def train(sess, task_config, nn, dataset):
    """
    進行訓練一輪
    """
    # 準備feed_dict需要的key
    feed_keys = list()
    feed_keys.extend(nn.input_keys())
    feed_keys.append(const.LABEL_GOLD)
    feed_keys = filter(lambda key: key not in {const.IS_TRAINING, const.CLASS_WEIGHTS}, feed_keys)

    loss = 0.
    labels_gold = list()
    labels_predict = list()

    # 初始化progressbar
    batch_index = 0
    pbar = progressbar(dataset.batch_num(task_config.batch_size))

    for subset_size, subset in dataset.batch_iterate(feed_keys, task_config.batch_size, shuffle=True, round_end=True):
        # 準備feed_dict
        feed_dict = dict()
        for key, placeholder in nn.ph_input.items():
            if key == const.IS_TRAINING:
                feed_dict[nn.get_input(const.IS_TRAINING)] = True
            elif key == const.CLASS_WEIGHTS:
                feed_dict[nn.get_input(const.CLASS_WEIGHTS)] = map(
                    lambda label: dataset.class_weight(label),
                    subset[const.LABEL_GOLD]
                )
            else:
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
        labels_gold.extend(subset[const.LABEL_GOLD])
        labels_predict.extend(partial_labels)

        # 更新progressbar
        batch_index += 1
        pbar.update(batch_index)
    pbar.finish()

    current_step = tf.train.global_step(sess, nn.global_step)
    loss /= dataset.n_sample

    labels_gold = labels_gold[:dataset.n_sample]
    labels_predict = labels_predict[:dataset.n_sample]
    return labels_gold, labels_predict, loss, current_step


def trial(sess, task_config, nn, dataset):
    """
    進行一輪驗證
    """
    # 準備feed_dict需要的key
    feed_keys = nn.input_keys()
    feed_keys.append(const.LABEL_GOLD)
    feed_keys = filter(lambda key: key not in {const.IS_TRAINING, const.CLASS_WEIGHTS}, feed_keys)

    loss = 0.
    labels_gold = list()
    labels_predict = list()

    # 初始化progressbar
    batch_index = 0
    pbar = progressbar(dataset.batch_num(task_config.batch_size))

    for subset_size, subset in dataset.batch_iterate(feed_keys, task_config.batch_size, shuffle=True, round_end=True):
        # 準備feed_dict
        feed_dict = dict()
        feed_dict[nn.dropout_keep_prob] = 1.
        for key, placeholder in nn.ph_input.items():
            if key == const.IS_TRAINING:
                feed_dict[nn.get_input(const.IS_TRAINING)] = True
            elif key == const.CLASS_WEIGHTS:
                feed_dict[nn.get_input(const.CLASS_WEIGHTS)] = map(
                    lambda label: dataset.class_weight(label),
                    subset[const.LABEL_GOLD]
                )
            else:
                feed_dict[placeholder] = subset[key]
        feed_dict[nn.label_gold] = subset[const.LABEL_GOLD]

        # 驗證
        partial_loss, partial_labels = sess.run(
            [nn.loss, nn.label_predict],
            feed_dict=feed_dict
        )

        # 更新本輪記錄
        loss += partial_loss * subset_size
        labels_gold.extend(subset[const.LABEL_GOLD])
        labels_predict.extend(partial_labels)

        # 更新progressbar
        batch_index += 1
        pbar.update(batch_index)
    pbar.finish()

    labels_gold = labels_gold[:dataset.n_sample]
    labels_predict = labels_predict[:dataset.n_sample]
    loss /= dataset.n_sample
    return labels_gold, labels_predict, loss


def test(sess, task_config, nn, dataset):
    feed_keys = nn.input_keys()
    labels_predict = list()

    # 準備feed_dict需要的key
    feed_keys = nn.input_keys()
    feed_keys = filter(lambda key: key not in {const.IS_TRAINING, const.CLASS_WEIGHTS}, feed_keys)

    # 初始化progressbar
    batch_index = 0
    pbar = progressbar(dataset.batch_num(task_config.batch_size))

    for subset_size, subset in dataset.batch_iterate(feed_keys, task_config.batch_size, shuffle=False, round_end=True):
        # 準備feed_dict
        feed_dict = dict()
        feed_dict[nn.dropout_keep_prob] = 1.
        for key, placeholder in nn.ph_input.items():
            if key == const.IS_TRAINING:
                feed_dict[nn.get_input(const.IS_TRAINING)] = True
            elif key == const.CLASS_WEIGHTS:
                pass
            else:
                feed_dict[placeholder] = subset[key]

        # 預測
        partial_labels = sess.run(
            nn.label_predict,
            feed_dict=feed_dict
        )
        labels_predict.extend(partial_labels)

        # 更新progressbar
        batch_index += 1
        pbar.update(batch_index)
    pbar.finish()

    labels_predict = labels_predict[:dataset.n_sample]
    return labels_predict
