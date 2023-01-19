import torch
import transformers as trf
from torch.nn import Linear, ReLU, Dropout

class BERTClassifier(torch.nn.Module):
    def __init__(self, model, num_labels=13, emb_size=768, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = model
        self.dropout = Dropout(dropout)
        self.fc1 = Linear(emb_size, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, num_labels)
        self.relu = ReLU()
        
    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = pooled_output[0]
        pooler = hidden_state[:, 0]
        pooler = self.dropout(pooler)
        pooler = self.fc1(pooler)
        pooler = self.relu(pooler)
        pooler = self.dropout(pooler)
        pooler = self.fc2(pooler)
        pooler = self.relu(pooler)
        pooler = self.dropout(pooler)
        output = self.fc3(pooler)
        return output