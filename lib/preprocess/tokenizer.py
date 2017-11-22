# -*- coding: utf-8 -*-
from nltk.tokenize import TweetTokenizer

_tokenizer = TweetTokenizer()


def tokenize(text):
    return _tokenizer.tokenize(text)
