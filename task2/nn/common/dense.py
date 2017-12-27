# -*- coding: utf-8 -*-
import tensorflow as tf


def build(ph_input, dim_output):
    w = tf.Variable(tf.truncated_normal([ph_input.shape[-1].value, dim_output], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[dim_output]))
    y = tf.matmul(ph_input, w) + b
    return y, w, b
