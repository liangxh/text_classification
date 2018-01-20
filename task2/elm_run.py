# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
from nltk import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import task2
from task2.lib import evaluate
import elm
import numpy as np


@commandr.command('run')
def run(key):
    """
    線性SVM
    輸入只用Tf-Idf
    """
    lexicon_feat = task2.dataset.load_lexicon_feature(key, 'train')
    labels_gold = task2.dataset.load_labels(key, 'train')
    labels_col = np.asarray(map(lambda l: [l], labels_gold))
    data = np.hstack([labels_col, lexicon_feat])

    elmk = elm.ELMKernel()
    elmk.search_param(data, cv="kfold", of="accuracy", eval=10)
    result = elmk.train(data)
    labels_predict = np.round(result.predicted_targets).astype(int)
    train_score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)

    lexicon_feat = task2.dataset.load_lexicon_feature(key, 'train')
    labels_gold = task2.dataset.load_labels(key, 'train')
    labels_col = np.asarray(map(lambda l: [l], labels_gold))
    data = np.hstack([labels_col, lexicon_feat])

    result = elmk.test(data)
    labels_predict = np.round(result.predicted_targets).astype(int)
    trial_score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)

    print('[TRAIN]', train_score_dict)
    print('[TRIAL]', trial_score_dict)


if __name__ == '__main__':
    commandr.Run()
