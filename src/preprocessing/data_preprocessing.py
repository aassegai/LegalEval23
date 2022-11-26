import pandas as pd
import numpy as np
import re
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from tqdm.auto import tqdm 

lemmatizer = WordNetLemmatizer()
label2id = {'PREAMBLE': 0,
            'FAC': 1,
            'RLC': 2,
            'ISSUE': 3,
            'ARG_PETITIONER': 4,
            'ARG_RESPONDENT': 5,
            'ANALYSIS': 6,
            'STA': 7,
            'PRE_RELIED': 8,
            'PRE_NOT_RELIED': 9,
            'RATIO': 10,
            'RPC': 11,
            'NONE': 12
}

id2label = {0: 'PREAMBLE',
            1: 'FAC',
            2: 'RLC',
            3: 'ISSUE',
            4: 'ARG_PETITIONER',
            5: 'ARG_RESPONDENT',
            6: 'ANALYSIS',
            7: 'STA',
            8: 'PRE_RELIED',
            9: 'PRE_NOT_RELIED',
            10: 'RATIO',
            11: 'RPC',
            12: 'NONE'
}


class DataPreprocessor:
    def __init__(self, lemmatizer=lemmatizer,
                 remove_punctuation=True,
                 lemmatize=True):
        self.lemmatizer = lemmatizer
        self.remove_punctuation = remove_punctuation
        self.lemmatize = lemmatize
    
    def filter_annotations(self, annotations):
        new_annotations = []
        for annotation in annotations:
            # print(annotation)
            new_result = []
            for result in annotation[0]['result']:
                if 'from_name' in result.keys():
                    result.pop('from_name')
                if 'to_name' in result.keys():
                    result.pop('to_name')            
                if 'type' in result.keys():
                    result.pop('type')
                new_result.append(result)
            new_result = {'result': new_result}
            new_annotations.append(new_result)
        return new_annotations

    def preprocess_text(self, texts, annotations):
        new_texts = []
        new_annotations = []
        for i in tqdm(range(len(texts))):
            temp_text = texts[i]['text'].strip().replace('\n', ' ').lower()
            if self.remove_punctuation:
                temp_text = re.sub(r'[^\w\s]', '', temp_text)
            prep_text = ''
            for word in temp_text.split():
                if self.lemmatize:
                    prep_text = prep_text + '' + self.lemmatizer.lemmatize(word) + ' '
                else:
                    prep_text = prep_text + '' + word + ' '
            new_texts.append(prep_text)


            # print(annotations[i]['result'])      
            segments = []
            for segment in annotations[i]['result']:
                # print(segment['value']['text'])
                temp_segment = segment['value']['text'].strip().replace('\n', ' ').lower()
                # print(segment['value']['text'])
                if self.remove_punctuation:
                    temp_segment = re.sub(r'[^\w\s]', '', temp_segment)
                prep_segment = ''
                for word in temp_segment.split():
                    if self.lemmatize:
                        prep_segment = prep_segment + '' + self.lemmatizer.lemmatize(word) + ' '
                    else: 
                        prep_segment = prep_segment + '' + word + ' '
                # print(prep_segment)
                segment['value']['start'] = prep_text.find(prep_segment)
                segment['value']['end'] = prep_text.find(prep_segment) + len(prep_segment)
                segment['value']['text'] = prep_segment
                segment['value']['labels'] = label2id[segment['value']['labels'][0]]
                segments.append(segment)

            new_annotations.append(segments)
            
        
        return new_texts, new_annotations



    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        self.dataframe = df
        self.dataframe['annotations'] = self.filter_annotations(df['annotations'])
        new_texts, new_annotations = self.preprocess_text(df.data, df.annotations)
        
        self.dataframe.drop_duplicates(subset='id', inplace=True)
        self.dataframe = self.dataframe.set_index('id')

        for i, idx in enumerate(self.dataframe.index):
            self.dataframe.loc[idx].data = new_texts[i]
            self.dataframe.loc[idx].annotations = new_annotations[i]
            if len(self.dataframe.loc[idx].annotations) == 0:
                self.dataframe.drop(idx, inplace=True)
        return self.dataframe
        
