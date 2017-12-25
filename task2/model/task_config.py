# -*- coding: utf-8 -*-
import os
import yaml
from task2.model.config import config


class TaskConfig(object):
    task_key = None
    embedding_algorithm = None
    embedding_key = None
    n_vocab = None
    dim_output = None
    dim_lexicon_feat = None
    dim_embed = None
    learning_rate_initial = 0.01
    learning_rate_decay_rate = 1.
    learning_rate_decay_steps = 10
    l2_reg_lambda = 0.2
    epochs = 5
    validate_interval = 1
    batch_size = 64
    seq_len = 50

    @classmethod
    def load(cls, filename):
        data = yaml.load(open(filename, 'r'))
        task_config = cls()
        for name, value in data.items():
            task_config.__setattr__(name, value)
        return task_config

    @property
    def dir_checkpoint(self):
        return os.path.join(config.dir_train_checkpoint, self.task_key)

    @property
    def prefix_checkpoint(self):
        return os.path.join(self.dir_checkpoint, 'model')


def _test():
    import os
    filename = os.path.join(os.environ['HOME'], 'lab', 'task2', 'config', 'sample.yaml')
    task_config = TaskConfig.load(filename)


if __name__ == '__main__':
    _test()
