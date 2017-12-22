# -*- coding: utf-8 -*-
import commandr
from task2 import dataset


@commandr.command('lexicon')
def build_lexicon_feat(key, mode):
    """
    生成lexicon feature，需要運行終端腳本，此處僅供提醒
    """
    print 'place execute the following command in the base'
    print 'bash ${HOME}' + '/lab/text_classification/task2/scripts/affective_tweets_feature.sh {} {}'.format(mode, key)


@commandr.command('tokenize')
def tokenize_text(key, mode):
    """
    為文本切詞，詞之前以空格隔開
    """
    from nlp.lib.preprocess.prepare import prepare_text

    text_list = dataset.load_texts(key, mode)
    filename_tokenized = dataset.path_to_tokenized(key, mode)
    with open(filename_tokenized, 'w') as file_obj:
        for text in text_list:
            token_list = prepare_text(text)
            file_obj.write(' '.join(token_list).encode('utf8') + '\n')


@commandr.command('vocab')
def build_vocab(key):
    """
    根據train數據集生成字典集報告
    """
    from nlp.lib.preprocess.vocab import VocabBuilder

    vocab_builder = VocabBuilder()
    tokenized = dataset.load_tokenized(key, 'train')
    for tokens in tokenized:
        vocab_builder.encounter(tokens)
    vocab_builder.export_report(dataset.path_to_vocab(key))


@commandr.command('subset')
def dataset_subset(in_key, out_key, n, mode, length=None):
    """
    生成數據集子集, 為便於調試神經網絡用
    """
    n = int(n)
    length = int(length) if length is not None else None

    labels = dataset.load_labels(in_key, mode)
    idx_list = list()
    label_count = dict([(i, 0) for i in range(n)])
    for i, label in enumerate(labels):
        if label < n:
            if length is None or label_count[label] < length:
                idx_list.append(i)
                label_count[label] += 1
    idx_set = set(idx_list)

    def transfer(path_to):
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

    transfer(dataset.path_to_labels)
    transfer(dataset.path_to_tokenized)
    transfer(dataset.path_to_lexicon_feat)


@commandr.command('statics')
def dataset_statics(key, mode):
    """
    給出數據集相關統計數據
    """
    max_seq_len = dataset.get_max_seq_len(key, mode)
    return max_seq_len


if __name__ == '__main__':
    commandr.Run()
