from transformers import BertModel
from torchcrf import CRF
import torch.nn as nn
import torch


class CustomBert(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, ids: torch.tensor, msk: torch.tensor, smax=False) -> torch.tensor:
        # bert_output is (batch_size, sequence_length, 768)
        bert_output = self.bert(ids, msk).last_hidden_state
        logits = self.linear(bert_output)
        if smax:
            logits = self.softmax(logits)
        return logits.transpose(1, 2)


class BertCRF(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, num_classes)
        self.crf = CRF(num_classes, batch_first=True)

    def forward(self, ids: torch.tensor, tags: torch.tensor, msk: torch.tensor) -> torch.tensor:
        tags = torch.where(tags == -100, 0, tags)
        msk = msk == 1
        # bert_output is (batch_size, sequence_length, 768)
        bert_output = self.bert(ids, msk).last_hidden_state
        emission = self.linear(bert_output)
        negative_log_likelihood = self.crf(emission, tags, mask=msk)
        return -1 * negative_log_likelihood

    def decode(self, ids: torch.tensor, msk: torch.tensor) -> torch.tensor:
        msk = msk == 1
        seq_output = self.bert(ids, msk).last_hidden_state
        emission = self.linear(seq_output)
        best_path = self.crf.decode(emission, mask=msk)
        return best_path
