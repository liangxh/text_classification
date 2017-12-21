# -*- coding: utf-8 -*-
import commandr
from lib.dataset import task2
from lib.preprocess.prepare import prepare_text
from lib.preprocess.vocab import VocabBuilder


@commandr.command('vocab')
def build_vocab(key):
    vocab_builder = VocabBuilder()
    text_list = task2.load_texts(key, 'train')
    for text in text_list:
        seq = prepare_text(text)
        vocab_builder.encounter(seq)
    vocab_builder.export_report(key)


@commandr.command('lexicon')
def build_lexicon_feat(key, mode):
    print 'place execute the following command in the base'
    print '$ bash PATH_TO_TASK2_SCRIPTS/affective_tweets_feature.sh {} {}'.format(mode, key)
    raise NotImplementedError()


@commandr.command('tokenize')
def tokenize_text(key, mode):
    text_list = task2.load_texts(key, mode)
    filename_tokenized = task2.path_to_tokenized(key, mode)
    with open(filename_tokenized, 'w') as file_obj:
        for text in text_list:
            token_list = prepare_text(text)
            file_obj.write(' '.join(token_list).encode('utf8') + '\n')


@commandr.command('subset')
def dataset_subset(in_key, out_key, n, mode, length=None):
    labels = task2.load_labels(in_key, mode)
    idx = list()
    label_count = dict([(i, 0) for i in range(n)])
    for i, label in enumerate(labels):
        if label < n:
            if length is None or label_count[label] < length:
                idx.append(i)
                label_count[label] += 1

    def transfer(path_to, in_key, out_key, mode, idx_set):
        in_filename = path_to(in_key, mode)
        out_filename = path_to(out_key, mode)
        with open(in_filename, 'r') as in_obj, open(out_filename, 'w') as out_obj:
            idx = 0
            hit_count = 0
            for line in in_obj:
                if idx in idx_set:
                    out_obj.write(line)
                    hit_count += 1
                    if hit_count == len(idx_set):
                        break
                idx += 1

        if not hit_count == len(idx_set):
            raise Exception('index mismatch')

    idx_set = set(idx)
    transfer(task2.path_to_labels, in_key, out_key, mode, idx_set)
    transfer(task2.path_to_tokenized, in_key, out_key, mode, idx_set)
    transfer(task2.path_to_lexicon_feat, in_key, out_key, mode, idx_set)


if __name__ == '__main__':
    commandr.Run()