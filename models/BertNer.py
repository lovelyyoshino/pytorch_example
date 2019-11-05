import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace
from .BasicModule import BasicModule
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert import BertForTokenClassification


class BertNer(BasicModule):
    def __init__(self, opt):
        super(BertNer, self).__init__()
        self.opt = opt
        self.num_labels = self.opt.tag_nums
        self.bertForToken = BertForTokenClassification.from_pretrained(self.opt.bert_model_dir, num_labels=self.opt.tag_nums)

    def forward(self, x, train=True):

        input_id, input_mask, segment_id, labels_id, ent_cnt = x
        sequence_output, _ = self.bertForToken.bert(input_id, token_type_ids=segment_id, attention_mask=input_mask, output_all_encoded_layers=False)
        sequence_output = self.bertForToken.dropout(sequence_output)  # (B, L, H)
        logits = self.bertForToken.classifier(sequence_output)

        if train:
            loss_fct = CrossEntropyLoss()
            loss_tags = loss_fct(logits.view(-1, self.num_labels), labels_id.view(-1))
            return loss_tags
        else:
            return logits
