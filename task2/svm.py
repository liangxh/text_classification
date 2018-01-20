# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
from nltk import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
import task2
from task2.lib import evaluate
from scipy import sparse
from collections import defaultdict
import math
from sklearn.feature_selection import SelectFromModel


@commandr.command('linear')
def linear(key, c=1., weight=1.):
    """
    線性SVM
    輸入只用Tf-Idf
    """
    class_weight = weight != 0

    c = float(c)
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words=None)

    X = vectorizer.fit_transform(task2.dataset.load_tokenized_as_texts(key, 'train'))
    labels_gold = task2.dataset.load_labels(key, 'train')

    if class_weight:
        class_count = defaultdict(lambda: 0.)
        for label in labels_gold:
            class_count[label] += 1
        class_weights = dict()
        for label, count in class_count.items():
            class_weights[label] = 1./count

        model = LinearSVC(C=c, verbose=1, class_weight=class_weights)
    else:
        model = LinearSVC(C=c, verbose=1)

    model.fit(X, labels_gold)
    labels_predict = model.predict(X)
    score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)
    print('[TRAIN]', score_dict)

    X = vectorizer.transform(task2.dataset.load_tokenized_as_texts(key, 'trial'))
    labels_gold = task2.dataset.load_labels(key, 'trial')
    labels_predict = model.predict(X)
    score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)
    print('[TRIAL]', score_dict)


@commandr.command('lex')
def lex(key, c=1.):
    """
    線性SVM
    輸入只用lexicon feature
    """
    c = float(c)

    lexicon_feat = task2.dataset.load_lexicon_feature(key, 'train')
    X = lexicon_feat

    labels_gold = task2.dataset.load_labels(key, 'train')
    model = LinearSVC(C=c, verbose=1)
    model.fit(X, labels_gold)
    labels_predict = model.predict(X)
    score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)
    print('[TRAIN]', score_dict)

    lexicon_feat = task2.dataset.load_lexicon_feature(key, 'trial')
    X = lexicon_feat

    labels_gold = task2.dataset.load_labels(key, 'trial')
    labels_predict = model.predict(X)
    score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)
    print('[TRIAL]', score_dict)


@commandr.command('llex')
def llex(key, c=1., weight=1.):
    """
    線性SVM
    輸入使用Tf-Idf + lexicon_feature
    """
    c = float(c)
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words=None)

    X = vectorizer.fit_transform(task2.dataset.load_tokenized_as_texts(key, 'train'))
    lexicon_feat = task2.dataset.load_lexicon_feature(key, 'train')
    X = sparse.hstack([X, lexicon_feat])

    labels_gold = task2.dataset.load_labels(key, 'train')

    class_weight = weight != 0
    if class_weight:
        class_count = defaultdict(lambda: 0.)
        for label in labels_gold:
            class_count[label] += 1
        class_weights = dict()
        for label, count in class_count.items():
            class_weights[label] = 1./count

        model = LinearSVC(C=c, verbose=1, class_weight=class_weights)
    else:
        model = LinearSVC(C=c, verbose=1)

    model.fit(X, labels_gold)
    labels_predict = model.predict(X)
    score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)
    print('[TRAIN]', score_dict)

    X = vectorizer.transform(task2.dataset.load_tokenized_as_texts(key, 'trial'))
    lexicon_feat = task2.dataset.load_lexicon_feature(key, 'trial')
    X = sparse.hstack([X, lexicon_feat])
    labels_gold = task2.dataset.load_labels(key, 'trial')
    labels_predict = model.predict(X)
    score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)
    print('[TRIAL]', score_dict)


@commandr.command('csupport')
def csupport(key, c=1., kernel='rbf'):
    """
    核SVM
    """
    c = float(c)
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words=None)

    X = vectorizer.fit_transform(task2.dataset.load_tokenized_as_texts(key, 'train'))
    labels_gold = task2.dataset.load_labels(key, 'train')
    model = SVC(C=c, kernel=kernel, verbose=1)
    model.fit(X, labels_gold)
    labels_predict = model.predict(X)
    score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)
    print('[TRAIN]', score_dict)

    X = vectorizer.transform(task2.dataset.load_tokenized_as_texts(key, 'trial'))
    labels_gold = task2.dataset.load_labels(key, 'trial')
    labels_predict = model.predict(X)
    score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)
    print('[TRIAL]', score_dict)


@commandr.command('stack')
def stack(key, c=1.):
    """
    線性SVM
    輸入只用Tf-Idf
    """
    c = float(c)
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words=None)

    X = vectorizer.fit_transform(task2.dataset.load_tokenized_as_texts(key, 'train'))
    labels_gold = task2.dataset.load_labels(key, 'train')

    model = LinearSVC(C=c, verbose=1)
    model.fit(X, labels_gold)
    feat_transformer = SelectFromModel(model, prefit=True)

    X = feat_transformer.transform(X)
    model = LinearSVC(C=c, verbose=1)
    model.fit(X, labels_gold)

    labels_predict = model.predict(X)
    train_score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)

    X = vectorizer.transform(task2.dataset.load_tokenized_as_texts(key, 'trial'))
    labels_gold = task2.dataset.load_labels(key, 'trial')

    X = feat_transformer.transform(X)
    labels_predict = model.predict(X)
    trial_score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)

    print('[TRAIN]', train_score_dict)
    print('[TRIAL]', trial_score_dict)


if __name__ == '__main__':
    commandr.Run()
