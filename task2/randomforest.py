# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
from nltk import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import task2
from task2.lib import evaluate
from scipy import sparse


@commandr.command('basic')
def basic(key, depth=10):
    """
    線性SVM
    輸入只用Tf-Idf
    """
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words=None)

    X = vectorizer.fit_transform(task2.dataset.load_tokenized_as_texts(key, 'train'))
    labels_gold = task2.dataset.load_labels(key, 'train')

    model = RandomForestClassifier(max_depth=depth, random_state=0, verbose=1)
    model.fit(X, labels_gold)

    labels_predict = model.predict(X)
    train_score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)

    X = vectorizer.transform(task2.dataset.load_tokenized_as_texts(key, 'trial'))
    labels_gold = task2.dataset.load_labels(key, 'trial')
    labels_predict = model.predict(X)
    trial_score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)
    print('[TRAIN]', train_score_dict)
    print('[TRIAL]', trial_score_dict)


@commandr.command('lex')
def lex(key, depth=10):
    """
    線性SVM
    輸入只用lexicon feature
    """
    lexicon_feat = task2.dataset.load_lexicon_feature(key, 'train')
    X = lexicon_feat

    labels_gold = task2.dataset.load_labels(key, 'train')

    model = RandomForestClassifier(max_depth=depth, random_state=0, verbose=1)
    model.fit(X, labels_gold)

    labels_predict = model.predict(X)
    train_score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)

    lexicon_feat = task2.dataset.load_lexicon_feature(key, 'trial')
    X = lexicon_feat

    labels_gold = task2.dataset.load_labels(key, 'trial')
    labels_predict = model.predict(X)
    trial_score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)
    print('[TRAIN]', train_score_dict)
    print('[TRIAL]', trial_score_dict)


@commandr.command('combo')
def combo(key, depth=10.):
    """
    線性SVM
    輸入使用Tf-Idf + lexicon_feature
    """
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words=None)

    X = vectorizer.fit_transform(task2.dataset.load_tokenized_as_texts(key, 'train'))
    lexicon_feat = task2.dataset.load_lexicon_feature(key, 'train')
    X = sparse.hstack([X, lexicon_feat])

    labels_gold = task2.dataset.load_labels(key, 'train')

    model = RandomForestClassifier(max_depth=depth, random_state=0, verbose=1)
    model.fit(X, labels_gold)

    labels_predict = model.predict(X)
    train_score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)

    X = vectorizer.transform(task2.dataset.load_tokenized_as_texts(key, 'trial'))
    lexicon_feat = task2.dataset.load_lexicon_feature(key, 'trial')
    X = sparse.hstack([X, lexicon_feat])
    labels_gold = task2.dataset.load_labels(key, 'trial')
    labels_predict = model.predict(X)
    trial_score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)
    print('[TRAIN]', train_score_dict)
    print('[TRIAL]', trial_score_dict)


if __name__ == '__main__':
    commandr.Run()
