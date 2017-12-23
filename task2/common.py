# -*- coding: utf-8 -*-
import copy
import numpy as np
import task2
from task2.model.config import config
from nlp.lib.word_embed.build import build_lookup_table, build_vocab_id_mapping


def input_list_to_batch(input_list, seq_len):
    # 在長度不足的輸入末尾补零
    input_batch = copy.deepcopy(input_list)
    for input_seq in input_batch:
        input_seq += [0] * (seq_len - len(input_seq))
    return input_batch


def load_embedding(key, task_config):
    # 加載詞嵌入相關數據
    vocab_list = task2.dataset.load_vocab(key, task_config.n_vocab)

    if task_config.embedding_algorithm == 'glove':
        from nlp.lib.word_embed.glove import Glove
        embedding = Glove(task_config.embedding_key)
    elif task_config.embedding_algorithm == 'word2vec':
        if task_config.embedding_key == 'frederic_godin':
            from nlp.lib.word_embed.word2vec.frederic_godin import FredericGodinModel
            embedding = FredericGodinModel(config.path_to_frederic_godin_index)
        else:
            raise Exception
    else:
        raise Exception

    task_config.dim_embed = embedding.dim

    # 加載詞嵌入相關模塊
    vocab_id_mapping = build_vocab_id_mapping(vocab_list)
    lookup_table = build_lookup_table(vocab_list, embedding)
    lookup_table = np.asarray(lookup_table)
    return vocab_id_mapping, lookup_table
