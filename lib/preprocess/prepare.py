# -*- coding: utf-8 -*-
import re
from lib.preprocess.tokenizer import tokenize

TOKEN_NUM = '<NUM>'
pattern_num = re.compile(r'\d+')
exclude_set = {u'\ufe0f', }


def prepare_text(text):
    if isinstance(text, str):
        text = text.decode('utf8')
    seq = tokenize(text)
    seq = map(unicode.lower, seq)
    tokens = list()
    for i, token in enumerate(seq):
        if pattern_num.match(token):
            tokens.append(TOKEN_NUM)
        elif token in exclude_set:
            pass
        else:
            tokens.append(token)
    return tokens


def _test():
    text = 'Happy Belated Birthday Love!!! @user 123 #HappyBirthdayJensenAckles #JensenAckles @user '
    print prepare_text(text)


if __name__ == '__main__':
    _test()
