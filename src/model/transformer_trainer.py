import numpy as np
import pandas as pd
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          BatchEncoding,
                          BertModel,
                          TrainingArguments,
                          Trainer)     
import evaluate
import torch
from torch import nn
from src.model.classification.gru_classifier import GRUClassifier

'''
This module is needed to unify all models training. 
The class TransformerTrainer loads the model 
and then fits it with given set of parameters and datasets using HF Trainer.
'''



class TransformerTrainer:
    def __init__(self, bert_name: str, 
                 num_labels:int, 
                 params: dict, 
                 id2label: dict, 
                 label2id: dict,
                 max_seq_len: int = 512, 
                 custom: bool = False):
        self.params = params
        self.bert_name = bert_name
        self.id2label = id2label
        self.label2id = label2id
        self.num_labels = num_labels
        self.max_seq_len = max_seq_len
        self.custom = custom

    def load_model(self):
        ''' A function to download the model from HF or build my own model 
             with 'custom' option. TODO: add support of an arbitrary torch model.
        '''
        tokenizer = AutoTokenizer.from_pretrained(self.bert_name,
                                                  use_cache=False)

        print("Loading :", self.bert_name)

        if self.custom:
            model = GRUClassifier(bert_name=self.bert_name,
                                          num_labels=self.num_labels,
                                          id2label = self.id2label,
                                          label2id = self.label2id)
            model.bert.config.eos_token_id = tokenizer.eos_token_id
            model.bert.config.pad_token_id = tokenizer.pad_token_id
            model.bert.config.cls_token_id = tokenizer.cls_token_id
            model.bert.config.max_new_tokens = model.bert.config.max_length = self.max_seq_len

        else:
            model = AutoModelForSequenceClassification.from_pretrained(self.bert_name,
                                                                      num_labels=self.num_labels,
                                                                      id2label=self.id2label,
                                                                      label2id=self.label2id,
                                                                      use_cache=False)
            model.config.eos_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.cls_token_id = tokenizer.cls_token_id
            model.config.max_new_tokens = model.config.max_length = self.max_seq_len

        return model, tokenizer

    def make_output_dir_name(self, custom_dir: str = None):
        # a function to make appropriate directory name for model saving

        if custom_dir is not None:
            return custom_dir + '/' + self.bert_name 
        else:  
            return './src/model/classification/' + self.bert_name
    

    def compute_metrics(self, eval_pred):

        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)

        precision_score = evaluate.load('precision')
        precision = precision_score.compute(predictions=preds, references=labels, average='weighted')['precision']
        recall_score = evaluate.load('recall')
        recall = recall_score.compute(predictions=preds, references=labels, average='weighted')['recall']    
        f1_score = evaluate.load('f1')
        weighted_f1 = f1_score.compute(predictions=preds, references=labels, average='weighted')['f1']

        return {
            'precison': precision,
            'recall': recall,
            'weighted_F1': weighted_f1,
        }

    
    def fit(self, 
            train_dataset, 
            val_dataset,
            save_model: bool=True):


        model, tokenizer = self.load_model()

        path = self.make_output_dir_name()
        training_args = TrainingArguments(
            output_dir=path,
            learning_rate=self.params['lr'],
            per_device_train_batch_size=self.params['batch_size'],
            per_device_eval_batch_size=self.params['batch_size'],
            num_train_epochs=self.params['n_epochs'],
            weight_decay=self.params['weight_decay'] if 'weight_decay' in self.params.keys() else 0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            eval_steps=1, 
            fp16=self.params['do_fp16'],  # maybe it will speed up the training a bit
            auto_find_batch_size=True, 
            dataloader_num_workers=self.params['num_workers'],
            optim = self.params['optimizer'] if 'optimizer' in self.params.keys() else 'adamw_torch', # avoids optimizer warnings
            full_determinism=True,
            logging_steps=100
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,
        )


        trainer.train()

        if save_model:
            trainer.save_model()

        # avoid out of memory errors
        del model

        return trainer


    def predict(self, test_dataset, trainer: Trainer):

        predictions, _, _ = trainer.predict(test_dataset)

        return predictions