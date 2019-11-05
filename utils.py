import os
import re
import codecs
import numpy as np
import time
import logging

def now():
    return str(time.strftime("%Y-%m-%d %H:%M:%S"))

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


import  matplotlib.pyplot as plt
def plot_epoch_for_performance_and_loss(path, res_dict):
    """Function: 评价指标以及训练集loss和epoch的关系曲线
    - param:
        path:存放图片的根目录
        model_name: (str) 模型的名称
        res_dict: (dict) 包含【key 模型名:model】
                             【key: 损失:epoch_loss】\【表现】随epoch变化的list
    """
    model = res_dict['model']
    out_dir = os.path.join(path, model)

    color = ['r', 'g', 'b', 'teal', 'y']
    shape = ['o', 'v', '^']
    ls = ['-', '--',':']
    loss = res_dict['epoch_loss']
    fig = plt.figure(figsize=(15,6)) # figsize指定给个图大小(两个数字分别表示横轴纵轴)
    ax1 = fig.add_subplot(1, 2, 1) # 1行2列的图，相当于四个图，1是第一个
    ax2 = fig.add_subplot(1, 2, 2) # 1行2列的图，相当于四个图，3是第三个
    ax1.plot(np.arange(len(loss)), np.array(loss))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    legend = []
    for idx, key in enumerate(list(res_dict.keys())[2:]):
        c = color[idx%len(color)]
        s = shape[idx%len(shape)]
        ax2.plot(np.arange(len(res_dict[key])), np.array(res_dict[key]), color=c, marker=s, linestyle=ls[idx%len(ls)])
        legend.append(key)
    ax2.legend(legend)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Perfermance")

    # plt.show()
    plt.savefig(out_dir+ '.png')
