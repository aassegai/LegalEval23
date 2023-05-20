import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset

label2id = {'PREAMBLE': 1,
            'FAC': 2,
            'RLC': 3,
            'ISSUE': 4,
            'ARG_PETITIONER': 5,
            'ARG_RESPONDENT': 6,
            'ANALYSIS': 7,
            'STA': 8,
            'PRE_RELIED': 9,
            'PRE_NOT_RELIED': 10,
            'RATIO': 11,
            'RPC': 12,
            'NONE': 0
}

id2label = {1: 'PREAMBLE',
            2: 'FAC',
            3: 'RLC',
            4: 'ISSUE',
            5: 'ARG_PETITIONER',
            6: 'ARG_RESPONDENT',
            7: 'ANALYSIS',
            8: 'STA',
            9: 'PRE_RELIED',
            10: 'PRE_NOT_RELIED',
            11: 'RATIO',
            12: 'RPC',
            0: 'NONE'
}


num_labels = 13
MAX_SEQUENCE_LENGTH = 512

class DatasetBuilder:
    def __init__(self, 
                 bert_name: str, 
                 label2id: dict = label2id,
                 id2label: dict = id2label,
                 MAX_SEQUENCE_LENGTH = 512
                 ):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name, use_cache=False)
        self.max_seq_len = MAX_SEQUENCE_LENGTH
        self.label2id = label2id
        self.id2label = id2label

    def tokenize_batch(self, batch):
        tokenized = self.tokenizer(batch['sentence'],
                            batch['context'],
                            padding="max_length",
                            truncation=True,  # truncate the longest
                            max_length=self.max_seq_len)

        return tokenized


    def build_dataset(self, dataframe: pd.DataFrame):

        # avoid possible confilct with other training
        df = dataframe.copy()

        print("Processing...")

        # remap labels to ids
        df['label'] = dataframe['label'].apply(lambda x: self.label2id[x])
        raw_dataset = Dataset.from_pandas(df)

        prepared_dataset = raw_dataset.map(
            self.tokenize_batch,
            batched=True,
            remove_columns=[col for col in raw_dataset.column_names if col != 'label']
        )

        return prepared_dataset