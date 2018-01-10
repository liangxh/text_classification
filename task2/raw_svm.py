# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
from nltk import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import task2
from task2.lib import evaluate


@commandr.command('run')
def run(key):
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words="english")
    # self.vectorizer = CountVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer,
    #                                   stop_words="english")

    X = vectorizer.fit_transform(task2.dataset.load_tokenized_as_texts(key, 'train'))
    labels_gold = task2.dataset.load_labels(key, 'train')
    # self.model = SVC(C=5000)
    model = LinearSVC(C=1)
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
