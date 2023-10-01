import os
import glob
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np
# from log import logger
import warnings
import pandas as pd
import numpy as np


os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

class_list = ['O']
bio2id = {}
id2bio = {}
for k in label2id.keys():
    b_key = 'B-' + k
    i_key = 'I-' + k
    class_list.extend([b_key, i_key])


for idx, key in enumerate(class_list):
    bio2id[key] = idx
    id2bio[idx] = key

class CoNLLDataset(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 train_data: bool = True,
                 max_instances: int = -1,
                 max_length: int = 512,
                 encoder_model: str = 'law-ai/InLegalBERT',
                 viterbi_algorithm: bool = True,
                 label_pad_token_id: int = -100,
                 label2id: dict = bio2id,
                 id2label: dict = id2bio, 
                 window_size = 2000
                 ):
        self.data = dataframe.copy()
        self.max_instances = max_instances
        self.max_length = max_length
        self.label_to_id = label2id
        self.id_to_label = id2label
        self.window_size = window_size
        self.encoder_model = encoder_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_model, padding_side='right')

        self.pad_token = self.tokenizer.special_tokens_map['pad_token']
        self.pad_token_id = self.tokenizer.get_vocab()[self.pad_token]
        self.sep_token = self.tokenizer.special_tokens_map['sep_token']
        self.instances = []
        self.sentences_words = []

        if viterbi_algorithm:
            self.label_pad_token_id = self.pad_token_id
        else:
            self.label_pad_token_id = label_pad_token_id
        self.read_data()




    def get_target_size(self):
        return len(set(self.label_to_id.values()))

    def get_target_vocab(self):
        return self.label_to_id

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def read_data(self):
        instance_idx = 0
        for idx, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            if self.max_instances != -1 and instance_idx > self.max_instances:
                break

            sentence_words, tags = row['tokens'], row['labels']
            ner_tags = []
            tokens = []
            if self.window_size and len(tags) > self.window_size:
               for start in range(0, len(tags) - 1, self.window_size // 2): # stride half window length
                    end = start + self.window_size - 2

                    window_tokens = sentence_words[start:end]
                    window_ner_tags = tags[start:end]

                    ner_tags.append(window_ner_tags)
                    tokens.append(window_tokens)


            for idx, tag in enumerate(ner_tags):
                tokenized_inputs = self.tokenizer(tokens[idx], truncation=True, padding='max_length',
                 is_split_into_words=True, max_length=self.window_size)
                input_ids = torch.tensor(tokenized_inputs['input_ids'], dtype=torch.long)
                labels = torch.tensor(self.tokenize_and_align_labels(tokenized_inputs, tag))
                attention_mask = torch.tensor(tokenized_inputs['attention_mask'], dtype=torch.bool)

                self.instances.append((input_ids, labels, attention_mask))
                self.sentences_words.append([tokens[idx], tag])
                instance_idx += 1

    # function from huggingface Token Classification ipynb
    # Set label for all tokens and -100 for padding and special tokens
    def tokenize_and_align_labels(self, tokenized_inputs, tags, label_all_tokens=True):
        previous_word_idx = None
        label_ids = []
        for word_idx in tokenized_inputs.word_ids():
            if word_idx is None:
                label_ids.append(self.label_pad_token_id)

            elif word_idx != previous_word_idx:
                label_ids.append(self.label_to_id[tags[word_idx]])
            else:
                label_ids.append(self.label_to_id[
                    tags[word_idx]] if label_all_tokens else self.label_pad_token_id)
            previous_word_idx = word_idx

        return label_ids


    def pad_instances(self, input_ids, labels, attention_masks):
        max_length_in_batch = max([len(token) for token in input_ids])
        input_ids_tensor = torch.empty(size=(len(input_ids), max_length_in_batch), dtype=torch.long).fill_(
            self.pad_token_id)
        labels_tensor = torch.empty(size=(len(input_ids), max_length_in_batch), dtype=torch.long).fill_(
            self.label_pad_token_id)
        attention_masks_tensor = torch.zeros(size=(len(input_ids), max_length_in_batch), dtype=torch.bool)

        for i in range(len(input_ids)):
            tokens_ = input_ids[i]
            seq_len = len(tokens_)

            input_ids_tensor[i, :seq_len] = tokens_
            labels_tensor[i, :seq_len] = labels[i]
            attention_masks_tensor[i, :seq_len] = attention_masks[i]

        return input_ids_tensor, labels_tensor, attention_masks_tensor

    def data_collator(self, batch):
        batch_ = list(zip(*batch))
        input_ids, labels, attention_masks = batch_[0], batch_[1], batch_[2]
        return self.pad_instances(input_ids, labels, attention_masks)


class SiameseDataset(CoNLLDataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 train_data: bool = True,
                 max_instances: int = -1,
                 max_length: int = 512,
                 encoder_model: str = 'law-ai/InLegalBERT',
                 viterbi_algorithm: bool = True,
                 label_pad_token_id: int = -100,
                 max_pairs: int = 100,
                 identical_entities_prob: float = 0.3,
                 search_iterations: int = 100,
                 label2id: dict = bio2id,
                 id2label: dict = id2bio
                 ):
        super(SiameseDataset, self).__init__(dataframe, train_data, max_instances, max_length,
                                             encoder_model, viterbi_algorithm, label_pad_token_id, label2id, id2label)
        self.data = dataframe

        self.max_pairs = max_pairs
        self.identical_entities_prob = identical_entities_prob
        self.search_iterations = search_iterations
        self.label_to_id = label2id
        self.id_to_label = id2label        

        self.paired_instances = []
        self.pairs_targets = []
        self.first_instances = []
        self.second_instances = []
        self.entities_in_data = {}

        self.parse_entities_in_data()
        self.entities = list(self.entities_in_data.keys())
        self.create_pairs()

    def __getitem__(self, item):
        return self.first_instances[item], self.second_instances[item], self.pairs_targets[item]

    def __len__(self):
        return len(self.paired_instances)

    def parse_entities_in_data(self):
        for sample_index, (input_ids, labels, _) in enumerate(self.instances):
            previous_label = ''
            for idx, label_id in enumerate(labels):
                label = self.id_to_label[label_id.item()]
                if label.startswith('B-') and label != previous_label:
                    if label not in self.entities_in_data.keys():
                        self.entities_in_data[label] = []
                    self.entities_in_data[label].append((sample_index, idx))
                previous_label = label

    def create_pairs(self):
        used_pairs = set()
        for _ in tqdm(range(self.max_pairs)):
            first_entity = np.random.choice(self.entities)
            idx = self.entities.index(first_entity)
            if np.random.random() > self.identical_entities_prob:
                second_entity = np.random.choice(self.entities[:idx] + self.entities[idx + 1:])
            else:
                second_entity = first_entity

            first_sample, second_sample = self.choice_two_pairs(first_entity, second_entity)

            if (first_sample, second_sample) in used_pairs:
                for _ in range(self.search_iterations):
                    first_sample, second_sample = self.choice_two_pairs(first_entity, second_entity)

            if (first_sample, second_sample) in used_pairs or first_sample == second_sample:
                warning = f'The pair {first_entity}-{second_entity} is not found.'
                warnings.warn(warning)
            else:
                first_input_ids, first_token_mask, first_attention_mask = self.parse_sample(first_entity, first_sample)
                second_input_ids, second_token_mask, second_attention_mask = self.parse_sample(second_entity, second_sample)

                pair_target = int(first_entity == second_entity)
                self.pairs_targets.append(pair_target)
                self.paired_instances.append((
                    [first_input_ids, second_input_ids],
                    [first_token_mask, second_token_mask],
                    [first_attention_mask, second_attention_mask], pair_target
                ))
                self.first_instances.append(
                    (first_input_ids, first_token_mask, first_attention_mask)
                )
                self.second_instances.append(
                    (second_input_ids, second_token_mask, second_attention_mask)
                )

                used_pairs.add((first_sample, second_sample))
                used_pairs.add((second_sample, first_sample))

    def choice_two_pairs(self, first_entity, second_entity):
        rng = np.random.default_rng()
        first_sample = tuple(rng.choice(self.entities_in_data[first_entity]))
        second_sample = tuple(rng.choice(self.entities_in_data[second_entity]))
        for _ in range(self.search_iterations):
            second_sample = tuple(rng.choice(self.entities_in_data[second_entity]))
            if first_sample != second_sample:
                break
        return first_sample, second_sample

    def parse_sample(self, entity, sample):
        sample_index, start_idx = sample
        input_ids, labels, attention_mask = self.instances[sample_index]
        token_mask = torch.zeros_like(labels)
        for idx, label in enumerate(labels[start_idx:]):
            if self.id_to_label[label.item()] not in [entity, entity.replace('B', 'I')]:
                break
            token_mask[start_idx+idx] = 1
        return input_ids, token_mask, attention_mask

    def data_collator(self, batch):
        batch_ = list(zip(*batch))
        first_instances, second_instances, pairs_targets = batch_[0], batch_[1], batch_[2]
        first_instances, second_instances = np.array(first_instances, dtype=object), np.array(second_instances, dtype=object)

        first_input_ids, first_token_mask, first_attention_mask = first_instances[:, 0], first_instances[:, 1], first_instances[:, 2]
        second_input_ids, second_token_mask, second_attention_mask = second_instances[:, 0], second_instances[:, 1], second_instances[:, 2]

        first_padded_instances = self.pad_instances(first_input_ids, first_token_mask, first_attention_mask)
        second_padded_instances = self.pad_instances(second_input_ids, second_token_mask, second_attention_mask)
        pairs_targets = torch.tensor(pairs_targets, dtype=torch.float)
        return first_padded_instances, second_padded_instances, pairs_targets
