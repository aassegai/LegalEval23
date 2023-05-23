import numpy as np
import pandas as pd
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          BatchEncoding,
                          BertModel,
                          TrainingArguments,
                          Trainer)     
import torch
from torch import nn

class GRUClassifier(nn.Module):
    def __init__(self, num_labels: int, bert_name: str, 
                       id2label: dict, label2id: dict, label_smoothing: int = 0.2, 
                       dropout: int = 0.3, max_seq_len:int=512):
        super(GRUClassifier, self).__init__()
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        self.bert = BertModel.from_pretrained(bert_name,
                                              num_labels=self.num_labels,
                                              id2label=id2label,
                                              label2id=label2id,
                                              use_cache=False)

        self.dropout = nn.Dropout(dropout)
        self.gru_1 = nn.GRU(self.bert.config.hidden_size, 256, batch_first=True, num_layers=3, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, self.num_labels)
        self.max_seq_len = max_seq_len
        self.pooling = nn.AvgPool1d(kernel_size=max_seq_len)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)


    def forward(self, input_ids, attention_mask, labels=None):
      
        hidden_state = self.bert(input_ids, attention_mask,
                                return_dict=False)
        
        output = self.dropout(hidden_state[0])


        # (batch_size, seq_len, hidden_size)
        gru_output, _ = self.gru_1(output)

        # (batch_size, hidden_size)
        gru_sent_emb = self.pooling(gru_output.reshape(gru_output.size[0], self.max_seq_len, self.bert.config.hidden_size)).reshape(gru_output.size[0], self.bert.config.hidden_size)

        # (1, batch_size, hidden_size)
        gru_sent_emb = torch.unsqueeze(gru_sent_emb, dim=0)
        gru_sent_emb = self.dropout2(gru_sent_emb)
        
        # (1, batch_size, num_labels)
        logits = self.classifier(gru_sent_emb)

        # (batch_size, num_labels))
        logits = torch.squeeze(logits, dim=0)

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits