# -*- coding: utf-8 -*-
import tensorflow as tf


def build(ph_input, dim_output, activation=None, bias=True):
    """
    由于tensorflow.layers.dense不會返回全連接層的 W和b，若L2 Loss有需要加入W和b時可以使用
    否則建議直接使用tensorflow.layers.dense
    """
    w = tf.Variable(tf.truncated_normal([ph_input.shape[-1].value, dim_output], stddev=0.1))
    y = tf.matmul(ph_input, w)

    if bias:
        b = tf.Variable(tf.constant(0.1, shape=[dim_output]))
        y += b

    if activation is not None:
        y = activation(y)

    if bias:
        return y, w, b
    else:
        return y, w


def batch_norm(ph_input, ph_training, dim_output, activation=None):
    last_state = tf.contrib.layers.fully_connected(ph_input, dim_output, activation_fn=None)
    last_state = tf.contrib.layers.batch_norm(last_state, center=True, scale=True, is_training=ph_training)
    if activation is not None:
        last_state = activation(last_state)
    return last_state
