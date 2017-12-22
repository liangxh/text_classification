# -*- coding: utf-8 -*-
import os

from nlp.lib.preprocess import vocab
from task2.model.config import config


def path_to_texts(key, mode):
    return os.path.join(config.dir_task2, mode, '{}.text'.format(key))


def path_to_labels(key, mode):
    return os.path.join(config.dir_task2, mode, '{}.labels'.format(key))


def path_to_tokenized(key, mode):
    return os.path.join(config.dir_task2, mode, '{}.tokenized'.format(key))


def path_to_lexicon_feat(key, mode):
    return os.path.join(config.dir_task2, mode, '{}.lexicon_feat'.format(key))


def path_to_vocab(key):
    return os.path.join(config.dir_task2, 'vocab', key)


def load_texts(key, mode):
    path = path_to_texts(key, mode)
    with open(path, 'r') as file_obj:
        content = file_obj.read().strip()
        content = content.decode('utf8')
        text_list = content.split('\n')

    return text_list


def load_labels(key, mode):
    path = path_to_labels(key, mode)
    with open(path, 'r') as file_obj:
        content = file_obj.read().strip()
        label_list = content.split('\n')
    label_list = map(int, label_list)
    return label_list


def load_tokenized(key, mode):
    path = path_to_tokenized(key, mode)
    with open(path, 'r') as file_obj:
        content = file_obj.read().strip()
        lines = content.decode('utf8').split('\n')
        tokenized_list = map(lambda line: line.split(' '), lines)
    return tokenized_list


def load_lexicon_feature(key, mode):
    path = path_to_lexicon_feat(key, mode)
    vecs = list()
    with open(path, 'r') as file_obj:
        for line in file_obj:
            line = line.strip()
            if line == '':
                continue
            vec = map(float, line.split('\t'))
            vecs.append(vec)
    return vecs


def load_vocab(key, n_vocab):
    path = path_to_vocab(key)
    return vocab.load(path, n_vocab)


def get_max_seq_len(key, mode):
    path = path_to_texts(key, mode)
    max_len = 0
    with open(path, 'r') as file_obj:
        for line in file_obj:
            max_len = max(max_len, len(line.split(' ')))
    return max_len


def get_output_dim(key):
    return max(*load_labels(key, 'train')) + 1


def get_lexicon_feature_dim(key):
    path = path_to_lexicon_feat(key, 'train')
    line = open(path, 'r').readline()
    dim = len(line.split('\t'))
    return dim
