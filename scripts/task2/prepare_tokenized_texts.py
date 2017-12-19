# -*- coding: utf-8 -*-
from lib.dataset import task2
from lib.preprocess.prepare import prepare_text
from optparse import OptionParser


def main():
    optparser = OptionParser()
    optparser.add_option('-k', dest='key')
    optparser.add_option('-M', dest='mode')
    opts, args = optparser.parse_args()

    text_list = task2.load_texts(opts.key, opts.mode)
    filename_tokenized = task2.path_to_tokenized(opts.key, opts.mode)
    with open(filename_tokenized, 'w') as file_obj:
        for text in text_list:
            token_list = prepare_text(text)
            file_obj.write(' '.join(token_list).encode('utf8') + '\n')


if __name__ == '__main__':
    main()
