# -*- coding: utf-8 -*-
from lib.dataset import task2
from lib.preprocess.prepare import prepare_text
from lib.preprocess.vocab import VocabBuilder
from optparse import OptionParser


def main():
    optparser = OptionParser()
    optparser.add_option('-k', dest='key')
    opts, args = optparser.parse_args()

    vocab_builder = VocabBuilder()
    text_list = task2.load_texts(opts.key, 'train')
    for text in text_list:
        seq = prepare_text(text)
        vocab_builder.encounter(seq)
    vocab_builder.export_report(opts.key)


if __name__ == '__main__':
    main()
