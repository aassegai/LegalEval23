import torch
from torch.utils.data import Dataset
import pandas as pd


class ClassificationDataset(Dataset):
    def __init__(self, tok_texts, labels, max_len):
        # self.data = dataframe
        self.texts = tok_texts
        # self.tokenizer = tokenizer
        self.targets = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        ids = self.texts['input_ids'][index]
        mask = self.texts['attention_mask'][index]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }