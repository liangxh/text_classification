# -*- coding: utf-8 -*-
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


def create(filename_config, score, time_mark):
    """
    將config和對應神經網絡源碼備份
    """
    task_config = TaskConfig.load(filename_config)

    dir_snapshot_task = os.path.join(dir_snapshot, task_config.task_key)
    if not os.path.exists(dir_snapshot_task):
        os.mkdir(dir_snapshot_task)

    dir_snapshot_run = os.path.join(
        dir_snapshot_task, '{:04d}_{}_{}'.format(
            int(score * 1000000), task_config.algorithm, time_mark
        )
    )
    os.mkdir(dir_snapshot_run)

    src_code = os.path.join(config.dir_task2_src, 'nn', '{}.py'.format(task_config.algorithm))
    dest_code = os.path.join(dir_snapshot_run, '{}.py'.format(task_config.algorithm))

    dest_config = os.path.join(dir_snapshot_run, filename_config.split('/')[-1])

    shutil.copy(src_code, dest_code)
    shutil.copy(filename_config, dest_config)
    print('task2.snapshot[INFO] snapshot created at {}'.format(dir_snapshot_run))

    notify_space(dir_snapshot_task, 1024)


def notify_space(dir_name, alert_size):
    """
    檢查該目錄的大小是否達到警告大小
    """
    output = subprocess.check_output(shlex.split('du -m -d 0 {}'.format(dir_name)))
    n_mb = int(output.split('\t')[0])
    if n_mb >= alert_size:
        print('task2.snapshot[INFO] {} has reached {} MB'.format(dir_name, n_mb))
