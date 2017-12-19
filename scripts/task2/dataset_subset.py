# -*- coding: utf-8 -*-
from optparse import OptionParser
from lib.dataset import task2


def main():
    optparser = OptionParser()
    optparser.add_option('-i', dest='in_key')
    optparser.add_option('-o', dest='out_key')
    optparser.add_option('-n', type=int, dest='top_n')
    optparser.add_option('-M', dest='mode')
    optparser.add_option('-l', type=int, dest='len')
    opts, args = optparser.parse_args()

    labels = task2.load_labels(opts.in_key, opts.mode)
    texts = task2.load_texts(opts.in_key, opts.mode)

    idx = list()
    label_count = dict([(i, 0) for i in range(opts.top_n)])
    for i, label in enumerate(labels):
        if label < opts.top_n:
            if label_count[label] < opts.len:
                idx.append(i)
                label_count[label] += 1

    subset_labels = map(lambda i: labels[i], idx)
    subset_texts = map(lambda i: texts[i], idx)

    path_texts = task2.path_to_texts(opts.out_key, opts.mode)
    path_labels = task2.path_to_labels(opts.out_key, opts.mode)

    with open(path_texts, 'w') as file_obj:
        file_obj.write(u'\n'.join(subset_texts).encode('utf8'))

    with open(path_labels, 'w') as file_obj:
        file_obj.write('\n'.join(map(str, subset_labels)))


if __name__ == '__main__':
    main()
