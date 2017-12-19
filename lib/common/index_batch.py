# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np


def generate(n, batch_size, shuffle=True):
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    idx = 0
    while idx < n:
        next_idx = idx + batch_size
        if next_idx <= n:
            yield indices[idx:next_idx]
        else:
            yield indices[idx:]
        idx = next_idx


def test():
    for indices in generate(11, 3, True):
        print(indices)


if __name__ == '__main__':
    test()
