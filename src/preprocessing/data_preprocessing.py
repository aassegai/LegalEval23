import pandas as pd
import numpy as np
import re
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from tqdm.auto import tqdm 

stopwords = nltk.corpus.stopwords.words('english')

lemmatizer = WordNetLemmatizer()


class DataPreprocessor:
    def __init__(self, lemmatizer=lemmatizer,
                 remove_punctuation=False,
                 lemmatize=False,
                 lower=False,
                 remove_stopwords=False):
        self.lemmatizer = lemmatizer
        self.remove_punctuation = remove_punctuation
        self.lemmatize = lemmatize
        self.lower = lower
        self.remove_stopwords = remove_stopwords
    
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
        for i in tqdm(texts.index):
            temp_text = texts[i]['text'].replace('\n', ' ').strip()
            if self.lower:
                temp_text = temp_text.lower()
            if self.remove_punctuation:
                temp_text = re.sub(r'[^\w\s]', ' ', temp_text)
            prep_text = ''
            for word in temp_text.split():
                if self.remove_stopwords:
                    if word not in stopwords:  
                        if self.lemmatize:
                            prep_text = prep_text + '' + self.lemmatizer.parse(word)[0].normal_form + ' '
                        else:
                            prep_text = prep_text + '' + word + ' '
                else:  
                    if self.lemmatize:
                        prep_text = prep_text + '' + self.lemmatizer.parse(word)[0].normal_form + ' '
                    else:
                        prep_text = prep_text + '' + word + ' '
            new_texts.append(prep_text)


            # print(annotations[i]['result'])      
            segments = []
            for segment in annotations[i]['result']:
                # print(segment['value']['text'])
                temp_segment = segment['value']['text'].strip().replace('\n', ' ')
                if self.lower:
                    temp_segment = temp_segment.lower()
                # print(segment['value']['text'])
                if self.remove_punctuation:
                    temp_segment = re.sub(r'[^\w\s]', ' ', temp_segment)
                prep_segment = ''
                for word in temp_segment.split():
                    if self.remove_stopwords:
                        if word not in stopwords:  
                            if self.lemmatize:
                                prep_segment = prep_segment + '' + self.lemmatizer.parse(word)[0].normal_form + ' '
                            else:
                                prep_segment = prep_segment + '' + word + ' '
                    else:  
                        if self.lemmatize:
                            prep_segment = prep_segment + '' + self.lemmatizer.parse(word)[0].normal_form + ' '
                        else:
                            prep_segment = prep_segment + '' + word + ' '            
                prep_segment = prep_segment.strip()
                segment['start'] = prep_text.find(prep_segment)
                segment['end'] = prep_text.find(prep_segment) + len(prep_segment)
                segment['text'] = prep_segment
                segment['label'] = segment['value']['labels']
                # segment['value']['labels'] = label2id[segment['value']['labels'][0]]
                segment.pop('value')
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
        
