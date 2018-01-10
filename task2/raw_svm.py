# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
from nltk import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
import task2
from task2.lib import evaluate


@commandr.command('linear')
def linear(key, c=1.):
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words=None)

    X = vectorizer.fit_transform(task2.dataset.load_tokenized_as_texts(key, 'train'))
    labels_gold = task2.dataset.load_labels(key, 'train')
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


@commandr.command('csupport')
def csupport(key, c=1., kernel='rbf'):
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


if __name__ == '__main__':
    commandr.Run()
