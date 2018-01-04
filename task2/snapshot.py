# -*- coding: utf-8 -*-
"""
snapshot只保存神經網絡的代碼和該次運行的config
理論上基於相同的代碼和config能复現結果
"""

from __future__ import print_function
import os
import shutil
import shlex
import subprocess
from task2.model.config import config
from task2.model.task_config import TaskConfig

dir_snapshot = os.path.join(config.dir_task2_src, 'snapshot')
if not os.path.exists(dir_snapshot):
    os.mkdir(dir_snapshot)


class Snapshot(object):
    def __init__(self, filename_config, time_mark):
        task_config = TaskConfig.load(filename_config)
        self.filename_config = filename_config
        self.algorithm = task_config.algorithm
        self.task_key = task_config.task_key
        self.time_mark = time_mark
        self.init_dir_snapshot_run = None

    @property
    def dir_snapshot_task(self):
        return os.path.join(dir_snapshot, self.task_key)

    def create(self):
        if not os.path.exists(self.dir_snapshot_task):
            os.mkdir(self.dir_snapshot_task)

        dir_snapshot_run = os.path.join(
            self.dir_snapshot_task, '{}_{}'.format(self.algorithm, self.time_mark)
        )
        os.mkdir(dir_snapshot_run)
        self.init_dir_snapshot_run = dir_snapshot_run

        src_code = os.path.join(config.dir_task2_src, 'nn', '{}.py'.format(self.algorithm))
        dest_code = os.path.join(dir_snapshot_run, '{}.py'.format(self.algorithm))
        dest_config = os.path.join(dir_snapshot_run, 'config.yaml')

        shutil.copy(src_code, dest_code)
        shutil.copy(self.filename_config, dest_config)
        print('task2.snapshot[INFO] snapshot created at {}'.format(dir_snapshot_run))

        notify_space(self.dir_snapshot_task, 1024)

    def rename_by_score(self, score):
        final_dir_snapshot_run = os.path.join(
            self.dir_snapshot_task, '{:06d}_{}_{}'.format(
                int(score * 1000000), self.algorithm, self.time_mark
            )
        )
        shutil.move(self.init_dir_snapshot_run, final_dir_snapshot_run)


def notify_space(dir_name, alert_size):
    """
    檢查該目錄的大小是否達到警告大小
    """
    output = subprocess.check_output(shlex.split('du -m -d 0 {}'.format(dir_name)))
    n_mb = int(output.split('\t')[0])
    if n_mb >= alert_size:
        print('task2.snapshot[INFO] {} has reached {} MB'.format(dir_name, n_mb))
