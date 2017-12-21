# -*- coding: utf-8 -*-
import math
import re


class TfIdf(object):
    def __init__(self, tf, df=1.):
        self.tf = tf
        self.df = df

    def update(self, inc_tf):
        self.tf += inc_tf
        self.df += 1

    def get(self, N):
        return self.tf * math.log(N / self.df)


class VocabBuilder(object):
    """
    統計數據集中各token的出現情況，生成報告
    """
    scorer_module = TfIdf

    def __init__(self):
        self.token_scorer = dict()
        self.doc_num = 0.
        self.doc_size = 0.

    def encounter(self, seq):
        self.doc_num += 1
        self.doc_size += len(seq)

        token_count = dict()
        for token in seq:
            if token in token_count:
                token_count[token] += 1
            else:
                token_count[token] = 1.

        for token, count in token_count.items():
            if token in self.token_scorer:
                self.token_scorer[token].update(count)
            else:
                self.token_scorer[token] = self.scorer_module(count)

    def export_report(self, filename):
        """
        導出字典統計報告
        """
        with open(filename, 'w') as file_obj:
            token_info_list = list()

            for token, scorer in self.token_scorer.items():
                score = scorer.get(self.doc_num)
                coverage = scorer.tf / self.doc_size
                token_info_list.append((token, score, scorer.tf, coverage))

            token_info_list = sorted(token_info_list, key=lambda k: -k[1])

            total_coverage = 0.
            for token, score, tf, coverage in token_info_list:
                total_coverage += coverage
                file_obj.write(u'{}\t{}\t{}\n'.format(token, int(tf), total_coverage).encode('utf8'))


def load(filename, n_vocab):
    """
    讀取字典報告中前n_vocab個詞
    """
    pattern_vocab = re.compile('^(\S+)\s*')
    vocab_list = list()
    with open(filename, 'r') as file_obj:
        for line in file_obj:
            res = pattern_vocab.search(line)
            if res is None:
                continue
            vocab = res.group(1)
            vocab_list.append(vocab)
            if n_vocab is not None and len(vocab_list) >= n_vocab:
                break
    vocab_list = map(lambda v: v.decode('utf8'), vocab_list)
    return vocab_list
