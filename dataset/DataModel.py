import codecs
import os
import random

import numpy as np
from texttable import Texttable
from ipdb import set_trace
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import Dataset


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, opt):
        self.opt = opt
        all_labels = self.get_labels()
        self.id2label = {i: j for i, j in enumerate(all_labels)}
        self.label2id = {j: i for i, j in enumerate(all_labels)}
        self.tokenization = BertTokenizer.from_pretrained(self.opt.vocab_dir)

    def get_train_examples(self):
        return self._create_example(
            self._read_data(os.path.join(self.opt.data_dir, "train"))
        )

    def get_dev_examples(self):
        return self._create_example(
            self._read_data(os.path.join(self.opt.data_dir, 'dev'))
        )

    def get_test_examples(self):
        return self._create_example(
            self._read_data(os.path.join(self.opt.data_dir, "test"))
        )

    def get_labels(self, data_dir=None):
        """
        此处填写代码
        """
        return ["O", "B-ORG", "I-ORG", "B-COMP", "I-COMP", "X"]

    def _read_data(self, input_file):
        """
        此处填写代码
        """
        return data

    def _create_single_example(self, param):
        """处理一个样例数据为数字表示
        param:
        return:
        """
        return x1, x2, x3

    def _create_example(self, lines):
        x1s = []
        x2s = []
        x3s = []
        for idx, (labels, text) in enumerate(lines):
            x1, x2, x3 = self._create_single_example(text, labels)
            x1s.append(x1)
            x2s.append(x2)
            x3s.append(x3)
            if idx == random.randint(0, 1000):
                print("Judge data...")
                print("************************")
                t = Texttable(max_width=self.opt.max_length)
                length = min(len(text), 24)   # 显示前20个字符即可
                t.add_row(['text:'] + text[:length])
                t.add_row(['labels:'] + labels[:length])
                t.add_row(['x1s:'] + x1s[:length])
                t.add_row(['x2s:'] + x2s[:length])
                t.add_row(['x3s: '] + list(x3s[:length]))
                print(t.draw())
                print("ent_cnt:", ent_cnt)
                print("************************")
        return list(zip(x1s, x2s, x3s)


class DataModel(Dataset):

    def __init__(self, opt, case='train'):
        self.opt = opt
        self.data_processer = DataProcessor(opt)
        self.x = self.load_data(case)
        # if case == 'train': self.x = self.x[:200]
        # set_trace()

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)

    def load_data(self, case):
        if case == 'train':
            return self.data_processer.get_train_examples()
        elif case == 'dev':
            return self.data_processer.get_dev_examples()

        return self.data_processer.get_test_examples()
