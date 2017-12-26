# -*- coding: utf-8 -*-
from task2.common import zero_padding
from task2.dataset import source_key_to_func
from task2.model.dataset import Dataset
from task2.model import const
from task2.nn.pack import NNPack


class BaseAlgorithm(object):
    feed_keys_input = [const.TOKEN_ID_SEQ, const.SEQ_LEN, const.LEXICON_FEAT]

    def __init__(self, task_config):
        self.config = task_config

    def load_and_prepare_dataset(self, mode, output=True, vocab_id_map=None):
        source_keys = [const.TOKEN_ID_SEQ, const.LEXICON_FEAT]
        if output:
            source_keys.append(const.LABEL_GOLD)

        # 讀取原始數據
        source_dict = dict(map(
            lambda key: (key, source_key_to_func[key](self.config.task_key, mode)),
            source_keys
        ))

        # 數據處理
        dataset = Dataset(source_dict)
        dataset.map(const.TOKEN_ID_SEQ, vocab_id_map)
        dataset.map(const.TOKEN_ID_SEQ, len, const.SEQ_LEN)
        dataset.map(const.TOKEN_ID_SEQ, zero_padding(self.config.seq_len))

        # 填补config
        self.config.dim_lexicon_feat = dataset.get_dim(const.LEXICON_FEAT)
        if output:
            self.config.dim_output = dataset.get_dim(const.LABEL_GOLD)

        return dataset

    @classmethod
    def _get_from_graph(cls, graph, keys):
        return dict(map(
            lambda key: (key, graph.get_operation_by_name(key).outputs[0]),
            keys
        ))

    @classmethod
    def build_from_graph(cls, graph):
        other_keys = [
            const.LABEL_PREDICT,
            const.LABEL_GOLD,
            const.DROPOUT_KEEP_PROB,
            const.LOSS,
            const.GLOBAL_STEP,
            const.OPTIMIZER
        ]
        ph_input = cls._get_from_graph(graph, cls.feed_keys_input)
        kwargs = cls._get_from_graph(graph, other_keys)

        return NNPack(ph_input=ph_input, **kwargs)
