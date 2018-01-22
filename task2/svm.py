# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
from nltk import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
import task2
from task2.lib import evaluate
from scipy import sparse
from sklearn.multiclass import OneVsRestClassifier


@commandr.command('linear')
def linear(key, c=1., lexicon=0.):
    """
    線性SVM
    輸入只用Tf-Idf
    """
    c = float(c)
    use_lexicon = float(lexicon) == 1
    print('use_lexicon: ', use_lexicon)

    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words=None)

    X = vectorizer.fit_transform(task2.dataset.load_tokenized_as_texts(key, 'train'))
    if use_lexicon:
        lexicon_feat = task2.dataset.load_lexicon_feature(key, 'train')
        X = sparse.hstack([X, lexicon_feat])

    labels_gold = task2.dataset.load_labels(key, 'train')

    model = LinearSVC(C=c, verbose=1)
    model.fit(X, labels_gold)
    labels_predict = model.predict(X)
    train_score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)

    X = vectorizer.transform(task2.dataset.load_tokenized_as_texts(key, 'trial'))
    if use_lexicon:
        lexicon_feat = task2.dataset.load_lexicon_feature(key, 'trial')
        X = sparse.hstack([X, lexicon_feat])

    labels_gold = task2.dataset.load_labels(key, 'trial')
    labels_predict = model.predict(X)
    trial_score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)

    print('[TRAIN]', train_score_dict)
    print('[TRIAL]', trial_score_dict)


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


@commandr.command('pca')
def stack(key, c=1., n_components=10):
    """
    線性SVM
    輸入只用Tf-Idf
    """
    from sklearn import decomposition
    c = float(c)
    n_components = int(n_components)
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words=None)
    pca = decomposition.SparsePCA(n_components=n_components)

    X = vectorizer.fit_transform(task2.dataset.load_tokenized_as_texts(key, 'train'))
    labels_gold = task2.dataset.load_labels(key, 'train')

    pca.fit(X)
    X = pca.transform(X)

    model = LinearSVC(C=c, verbose=1)
    model.fit(X, labels_gold)

    labels_predict = model.predict(X)
    train_score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)

    X = vectorizer.transform(task2.dataset.load_tokenized_as_texts(key, 'trial'))
    X = pca.transform(X)
    labels_gold = task2.dataset.load_labels(key, 'trial')

    labels_predict = model.predict(X)
    trial_score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)

    print('[TRAIN]', train_score_dict)
    print('[TRIAL]', trial_score_dict)


@commandr.command('ovr')
def one_vs_rest(key, c=1.):
    """
    線性SVM
    輸入只用Tf-Idf
    """
    c = float(c)
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words=None)

    X = vectorizer.fit_transform(task2.dataset.load_tokenized_as_texts(key, 'train'))
    labels_gold = task2.dataset.load_labels(key, 'train')

    model = OneVsRestClassifier(LinearSVC(C=c, verbose=1))
    model.fit(X, labels_gold)

    labels_predict = model.predict(X)
    train_score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)

    X = vectorizer.transform(task2.dataset.load_tokenized_as_texts(key, 'trial'))
    labels_gold = task2.dataset.load_labels(key, 'trial')

    labels_predict = model.predict(X)
    trial_score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)

    print('[TRAIN]', train_score_dict)
    print('[TRIAL]', trial_score_dict)


@commandr.command('knn')
def knn(key, n_neighbors=15, weights='uniform', lexicon=0.):
    from sklearn.neighbors import KNeighborsClassifier

    use_lexicon = float(lexicon) == 1.
    print('use_lexicon: ', use_lexicon)

    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words=None)

    X = vectorizer.fit_transform(task2.dataset.load_tokenized_as_texts(key, 'train'))

    if use_lexicon:
        lexicon_feat = task2.dataset.load_lexicon_feature(key, 'train')
        X = sparse.hstack([X, lexicon_feat])

    labels_gold = task2.dataset.load_labels(key, 'train')

    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    model.fit(X, labels_gold)
    labels_predict = model.predict(X)
    train_score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)

    X = vectorizer.transform(task2.dataset.load_tokenized_as_texts(key, 'trial'))
    if use_lexicon:
        lexicon_feat = task2.dataset.load_lexicon_feature(key, 'trial')
        X = sparse.hstack([X, lexicon_feat])

    labels_gold = task2.dataset.load_labels(key, 'trial')
    labels_predict = model.predict(X)
    trial_score_dict = evaluate.score(labels_predict=labels_predict, labels_gold=labels_gold)

    print('[TRAIN]', train_score_dict)
    print('[TRIAL]', trial_score_dict)


if __name__ == '__main__':
    commandr.Run()
