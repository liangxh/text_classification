# -*- coding: utf-8 -*-
import json
import commandr
import task2
from task2.model.config import config


@commandr.command('fg')
def build_frederic_godin_index(key, n):
    from nlp.lib.word_embed.word2vec.frederic_godin import FredericGodinModel

    vocab_list = task2.dataset.load_vocab(key, n)
    index = FredericGodinModel.build_specific_index(vocab_list)
    json.dumps(index, open(config.path_to_frederic_godin_index(key), 'w'))


if __name__ == '__main__':
    commandr.Run()
