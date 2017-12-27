# -*- coding: utf-8 -*-
import os
import yaml
from task2.model.config import config


class TaskConfig(object):
    task_key = None
    algorithm = None
    embedding_algorithm = None
    embedding_key = None
    n_vocab = None
    dim_output = None
    dim_lexicon_feat = None
    dim_embed = None
    learning_rate_initial = None
    learning_rate_decay_rate = None
    learning_rate_decay_steps = None
    l2_reg_lambda = None
    epochs = None
    validate_interval = None
    batch_size = None
    seq_len = None
    nickname = None

    @classmethod
    def load(cls, filename):
        data = yaml.load(open(filename, 'r'))
        task_config = cls()
        for name, value in data.items():
            task_config.__setattr__(name, value)
        return task_config

    @property
    def dir_checkpoint(self):
        return os.path.join(
                config.dir_train_checkpoint,
                self.task_key,
                self.nickname if self.nickname is not None else self.algorithm
            )

    @property
    def prefix_checkpoint(self):
        return os.path.join(self.dir_checkpoint, 'model')


def _test():
    import os
    filename = os.path.join(os.environ['HOME'], 'lab', 'task2', 'config', 'sample.yaml')
    task_config = TaskConfig.load(filename)


if __name__ == '__main__':
    _test()
