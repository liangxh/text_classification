# -*- coding: utf-8 -*-
import numpy as np
import task2
from task2.model.config import config
from nlp.lib.word_embed.build import build_lookup_table, build_vocab_id_mapping


def zero_padding(seq_len):
    def _pad(seq):
        seq += [0] * (seq_len - len(seq))
        return seq
    return _pad


def load_embedding(task_config, return_lookup_table=True):
    # 加載詞嵌入相關數據
    vocab_list = task2.dataset.load_vocab(task_config.task_key, task_config.n_vocab)
    vocab_id_mapping = build_vocab_id_mapping(vocab_list)

    if return_lookup_table:
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
        lookup_table = build_lookup_table(vocab_list, embedding)
        lookup_table = np.asarray(lookup_table)

        return vocab_id_mapping, lookup_table
    else:
        return vocab_id_mapping

