import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace
from .BasicModule import BasicModule
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert import BertForTokenClassification
from torchcrf import CRF

class BertCrfNer(BasicModule):
    def __init__(self, opt):
        super(BertCrfNer, self).__init__()
        self.opt = opt
        self.num_labels = self.opt.tag_nums
        self.bertForToken = BertForTokenClassification.from_pretrained(self.opt.bert_model_dir, num_labels=self.opt.tag_nums)
        self.crf = CRF(self.opt.tag_nums, batch_first=True)

    def forward(self, x, train=True):

        input_id, input_mask, segment_id, labels_id, ent_cnt = x
        sequence_output, _ = self.bertForToken.bert(input_id, token_type_ids=segment_id, attention_mask=input_mask, output_all_encoded_layers=False)
        sequence_output = self.bertForToken.dropout(sequence_output)  # (B, L, H)
        logits = self.bertForToken.classifier(sequence_output)

        if train:
            return -self.crf(logits, labels_id, input_mask.byte())
            # return -self.crf(logits, labels_id)
        else:
            return self.crf.decode(logits, input_mask.byte())
            # return self.crf.decode(logits)
