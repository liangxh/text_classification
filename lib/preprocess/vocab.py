# -*- coding: utf-8 -*-
import math


class TfIdf(object):
    def __init__(self, tf, df=1.):
        self.tf = tf
        self.df = df

    def update(self, inc_tf):
        self.tf += inc_tf
        self.df += 1

    def get(self, N):
        return self.tf * math.log(N / self.df)


class VocabularyBuilder(object):
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
        with open(filename, 'r') as file_obj:
            token_info_list = list()

            for token, scorer in self.token_scorer.items():
                score = scorer.get(self.doc_num)
                coverage = tfidf.tf / self.doc_size
                token_info_list.append((token, score, coverage))

            token_info_list = sorted(token_info_list, key=lambda k: -k[1])

            total_coverage = 0.
            for token, score, coverage in token_info_list:
                total_coverage += coverage
                file_obj.write('{}\t{}\t{}\t{}\n'.format(token, score, coverage, total_coverage))
