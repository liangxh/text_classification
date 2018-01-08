# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
import math
import numpy as np
from scipy import sparse
from sklearn.svm import LinearSVC
from progressbar import ProgressBar, Percentage, Bar, ETA
import task2
from task2.lib import evaluate
from collections import defaultdict


def progressbar(maxval):
    widgets = ['Progress: ', Percentage(), ' ', Bar(marker='>'), ' ', ETA()]
    return ProgressBar(widgets=widgets, maxval=maxval).start()


class TfIdf(object):
    def __init__(self, tf, df=1.):
        self.tf = tf
        self.df = df

    def update(self, inc_tf):
        self.tf += inc_tf
        self.df += 1

    def get(self, N):
        return self.tf * math.log(N / self.df)


class TfIdfTransformer(object):
    """
    統計數據集中各token的出現情況，生成報告
    """
    scorer_module = TfIdf

    def __init__(self):
        self.token_scorer = dict()
        self.doc_num = 0.
        self.doc_size = 0.
        self.dim = None
        self.token_idx = None
        self.idf_vec = None

    def encounter(self, seq):
        self.doc_num += 1
        self.doc_size += len(seq)

        token_count = dict()
        for token in seq:
            if token in token_count:
                token_count[token] += 1
            else:
                token_count[token] = 1.

        for token, count in token_count.items():
            if token in self.token_scorer:
                self.token_scorer[token].update(count)
            else:
                self.token_scorer[token] = self.scorer_module(count)

    def init_vec_config(self, dim):
        dim = min(dim, len(self.token_scorer))

        token_score = list()
        for token, scorer in self.token_scorer.items():
            score = scorer.get(self.doc_num)
            token_score.append((token, score))
        token_score = sorted(token_score, key=lambda item: -item[1])

        idf_vec = np.zeros(dim)
        token_idx = dict()

        for idx, item in enumerate(token_score[:dim]):
            token, score = item
            token_idx[token] = idx
            idf_vec[idx] = score

        self.dim = dim
        self.idf_vec = idf_vec
        self.token_idx = token_idx

    def to_vec(self, seq):
        tf_vec = np.zeros(self.dim)

        tf = defaultdict(lambda: 0)
        for token in seq:
            tf[self.token_idx.get(token, None)] += 1

        vec = np.zeros(self.dim)
        for idx, f in tf.items():
            if idx is None:
                continue
            vec[idx] = f * self.idf_vec[idx]

        vec_sum = np.sum(vec)
        if vec_sum > 0:
            vec /= vec_sum
        return vec


@commandr.command('run')
def run(key, dim=100000):
    dim = int(dim)

    def build_transformer():
        transformer = TfIdfTransformer()
        train_tokenized = task2.dataset.load_tokenized(key, 'train')

        print('calculating tf-idf')
        map(transformer.encounter, train_tokenized)
        transformer.init_vec_config(dim)
        return transformer

    transformer = build_transformer()

    def load(mode):
        tokenized = task2.dataset.load_tokenized(key, mode)
        labels = task2.dataset.load_labels(key, mode)

        batch_index = 0
        pbar = progressbar(len(tokenized))

        vecs = list()
        for tokens in tokenized:
            vec = transformer.to_vec(tokens)
            vec = sparse.csr.csr_matrix(vec)
            vecs.append(vec)

            # 更新progressbar
            batch_index += 1
            pbar.update(batch_index)
        pbar.finish()

        vecs = sparse.vstack(vecs)
        return vecs, labels

    print('building vectors')
    train_input, train_labels = load('train')
    trial_input, trial_labels = load('trial')

    print('start learning..')
    model = LinearSVC(verbose=1)
    model.fit(train_input, train_labels)

    labels_predict = model.predict(train_input)
    score_dict = evaluate.score(train_labels, labels_predict)
    print('[TRAIN]', score_dict)

    labels_predict = model.predict(trial_input)
    score_dict = evaluate.score(trial_labels, labels_predict)
    print('[TRIAL]', score_dict)


@commandr.command('run_lex')
def run_lex(key, dim=100000):
    dim = int(dim)

    transformer = TfIdfTransformer()
    train_tokenized = task2.dataset.load_tokenized(key, 'train')
    train_labels = task2.dataset.load_labels(key, 'train')
    train_lexicon = task2.dataset.load_lexicon_feature(key, 'train')

    trial_tokenized = task2.dataset.load_tokenized(key, 'trial')
    trial_labels = task2.dataset.load_labels(key, 'trial')
    trial_lexicon = task2.dataset.load_lexicon_feature(key, 'trial')

    print('calculating tf-idf')
    map(transformer.encounter, train_tokenized)
    transformer.init_vec_config(dim)

    print('building vectors')
    train_vecs = map(transformer.to_vec, train_tokenized)
    train_input = map(lambda v1_v2: np.concatenate(v1_v2), zip(train_vecs, train_lexicon))

    trial_vecs = map(transformer.to_vec, trial_tokenized)
    trial_input = map(lambda v1_v2: np.concatenate(v1_v2), zip(trial_vecs, trial_lexicon))

    print('start learning..')
    model = LinearSVC(verbose=1)
    model.fit(train_input, train_labels)

    labels_predict = model.predict(train_input)
    score_dict = evaluate.score(train_labels, labels_predict)
    print('[TRAIN]', score_dict)

    labels_predict = model.predict(trial_input)
    score_dict = evaluate.score(trial_labels, labels_predict)
    print('[TRIAL]', score_dict)


if __name__ == '__main__':
    commandr.Run()
