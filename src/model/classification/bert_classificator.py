import torch

class BERTClassificator(torch.nn.Module):
    def __init__(self, model, num_labels=13, emb_size=768, dropout=0.3):
        super(BERTClassificator, self).__init__()
        self.l1 = model
        self.pre_classifier = torch.nn.Linear(emb_size, emb_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(emb_size, num_labels)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output