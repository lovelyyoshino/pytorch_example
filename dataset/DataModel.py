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
            return ["O", "B-ORG", "I-ORG", "B-COMP", "I-COMP", "X"]

    def _read_data(self, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contents = line.strip()
                tokens = contents.split()
                if contents.startswith("-DOCSTART-"):
                    continue
                if len(tokens) >= 2:
                    words.append(tokens[0])
                    labels.append(tokens[-1])
                    if len(labels) > self.opt.max_length:
                        set_trace()
                else:
                    if len(contents) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words = []
                        labels = []
            return lines

    def _create_single_example(self, text, labels):
        """处理一个样例数据
        param
            - text: 一个句子
            - labels: 句子的标签
        return:
            - input_id:   (list) 用单词id表示的句子, 固定长度，不足补0
            - input_mask: (list) 0/1: 表示是否是补的字
            - segment_id: (list) 全为0
            - labels_ids: (list) 用数字表示的字的
            - ent_cnt: (0/1) 是否有实体
        """
        ent_cnt = int('B' in labels)

        label_list = labels.split()
        text_list = text.split()

        text_remove_unicode = []
        label_remove_unicode = []

        input_id = []
        labels_ids = []
        assert len(label_list) == len(text_list)

        for char, label in zip(text_list, label_list):
            # set_trace()
            char = self.tokenization.tokenize(char)
            if len(char) == 0:
                continue
            char_id = self.tokenization.convert_tokens_to_ids(char)
            assert len(char_id) == 1
            input_id.append(char_id[0])
            labels_ids.append(self.label2id[label])

            text_remove_unicode.append(char[0])
            label_remove_unicode.append(label)

        input_mask = np.zeros(self.opt.max_length).astype(int)
        segment_id = np.zeros(self.opt.max_length).astype(int)

        length = len(input_id)
        input_mask[: length] = 1

        add_num = max(0, self.opt.max_length - length)
        input_id = input_id + [0] * add_num
        labels_ids = labels_ids + [0] * add_num

        assert len(input_id) == len(input_mask) and len(input_id) == len(segment_id) \
            and len(input_id) == len(labels_ids)

        return text_remove_unicode, label_remove_unicode, input_id, input_mask, segment_id, labels_ids, ent_cnt

    def _create_example(self, lines):
        texts = []
        input_idxs = []
        input_masks = []
        segment_ids = []
        labels_ids = []
        ent_cnts = []
        for idx, (labels, text) in enumerate(lines):
            labels = 'O ' + labels
            text = '[CLS] ' + text

            text, labels, input_id, input_mask, segment_id, labels_id, ent_cnt = self._create_single_example(text, labels)
            texts.append(text)
            input_idxs.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            labels_ids.append(labels_id)
            ent_cnts.append(ent_cnt)
            if idx == random.randint(0, 1000):
                print("Judge data...")
                print("************************")
                t = Texttable(max_width=self.opt.max_length)
                length = min(len(text), 24)   # 显示前20个字符即可
                t.add_row(['text:'] + text[:length])
                t.add_row(['intput_id:'] + input_id[:length])
                t.add_row(['labels:'] + labels[:length])
                t.add_row(['labels_id:'] + labels_id[:length])
                t.add_row(['input_mask:'] + list(input_mask[:length]))
                t.add_row(['segment_id: '] + list(segment_id[:length]))
                print(t.draw())
                print("ent_cnt:", ent_cnt)
                print("************************")
        return list(zip(texts, np.array(input_idxs), np.array(input_masks), np.array(segment_ids), np.array(labels_ids), np.array(ent_cnts)))


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
