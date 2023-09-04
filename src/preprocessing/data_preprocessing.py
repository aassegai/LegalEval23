import pandas as pd
import numpy as np
import re
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from tqdm.auto import tqdm 
from transformers import AutoTokenizer

stopwords = nltk.corpus.stopwords.words('english')

lemmatizer = WordNetLemmatizer()

class DataPreprocessor:

    def __init__(self, lemmatizer=lemmatizer,
                 remove_punctuation=False,
                 lemmatize=False,
                 lower=False,
                 remove_stopwords=False):
        '''
        Given the raw data build more comfortable data structure 
        and transforms the text in it with options below, 
        returns the preprocessed dataframe.
        Options:
        - text decapitalization (transforming to lower case)
        - punctuation removal
        - stopwords removal
        - word normalization
        '''
        self.lemmatizer = lemmatizer
        self.remove_punctuation = remove_punctuation
        self.lemmatize = lemmatize
        self.lower = lower
        self.remove_stopwords = remove_stopwords
    
    def filter_annotations(self, annotations):
        '''
        Removes unnecessary data from annotations.
        '''
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
        '''
        Given the text and annotations preprocesses it in parallel.
        '''
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






class ContextExtractor:
    def __init__(self, max_context_len=512):
        self.max_context_len = max_context_len

    def get_token_level_idx(self, text: str,
                   char_index: int) -> int:
        """
          Converts the given char_index its equivalent token level index 
          in the given text.
        """
        if char_index == 0:
            return 0

        # Find the previous and next spaces around the given index
        previous_space_idx = text[:char_index].rfind(' ')
        next_space_idx = text[char_index:].find(' ')

        # If the previous space is found
        if previous_space_idx != -1:
            # Count the number of words before the current word
            token_level_index = len(text[:previous_space_idx].split())
            return token_level_index

        # If only the next space is found
        elif next_space_idx != -1:
            # Count the number of words before the next word
            token_level_index = len(text[:char_index].split())
            return token_level_index - 1

        # If no space is found
        else:
            # Count the number of words before the next word
            token_level_index = len(text[:char_index].split())
            return token_level_index


    def extract_context(self, text: str,
                                 sentence: str,
                                 span_start: int,
                                 span_end: int) -> str:
        """
          Given the text substring, it extracts the context of the sentence
          of given length from the text.
        """

        context = []

        span_start = self.get_token_level_idx(text, span_start)
        span_end = self.get_token_level_idx(text, span_end)

        text = text.split()
        sentence = sentence.split()
        sentence_len = len(sentence)

         
        window_len = int((self.max_context_len - (sentence_len * 2)) / 2)

        if window_len <= 0:
            return " ".join(context)

        if span_start <= 0:
            context += sentence

            idx = span_end + 1
            while idx <= span_end + window_len * 2:
                context.append(text[idx])
                idx += 1

            return " ".join(context)

        if span_end >= len(text):
            idx = span_start - window_len * 2
            while idx < span_start:
                context.append(text[idx])
                idx += 1

            context += sentence

            return " ".join(context)

        if span_start < window_len:
            idx = 0
            while idx < span_start:
                context.append(text[idx])
                idx += 1

            context += sentence

            idx = span_end + 1
            while idx <= span_end + window_len + (window_len - span_start):
                context.append(text[idx])
                idx += 1

            return " ".join(context)

        if window_len > (len(text) - span_end):
            idx = span_start - window_len - (window_len - (len(text) - span_end))
            while idx < span_start:
                context.append(text[idx])
                idx += 1

            context += sentence

            idx = span_end + 1
            while idx <= len(text) - 1:
                context.append(text[idx])
                idx += 1

            return " ".join(context)

        idx = span_start - window_len
        while idx < span_start:
            context.append(text[idx])
            idx += 1

        context += sentence

        idx = span_end + 1
        while idx < span_end + window_len:
            context.append(text[idx])
            idx += 1

        return " ".join(context)


    def __call__(self, data: pd.DataFrame, for_test=False) -> pd.DataFrame:
        """
          Given the preprocessed data transforms it into the 
          set of sentences needed to classify with their context
        """

        columns = ['doc_id', 'text', 'context', 'sentence', 'label']

        new_dataset = []

        print('Extracting context...')
        for idx in tqdm(data.index):
            whole_text = data.loc[idx]['data']
            for annotation in data.loc[idx]['annotations']:
                annotation_text = annotation['text']
                annotation_start = annotation['start']
                annotation_end = annotation['end']
                annotation_label = annotation['label']

                row = [idx, 
                       whole_text,
                       self.extract_context(whole_text, annotation_text, 
                       annotation_start, annotation_end), 
                       annotation_text,
                       annotation_label[0]
                       ]


                new_dataset.append(row)

        new_dataset = pd.DataFrame(new_dataset, columns=columns)
        if not for_test:
            new_dataset.drop_duplicates(['text', 'sentence'], inplace=True)

        return new_dataset

default_tokenizer = 'law-ai/InLegalBERT'
class BIOTagger():

    def __init__(self, tokenizer_name=default_tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True, use_fast=True)


    def adapt_indexes_without_spaces(self, row):
        """
        This function is very similar to the clean data one, it basically removes any possible space and adapt the start/end indexes consequently.
        We need this new start/end since the labeling step will work character wise, so we don't want to count spaces.
        """
        start = [annotation['start'] for annotation in row.annotations]
        end = [annotation['end'] for annotation in row.annotations]
        context = row.data
        text = [annotation['text'] for annotation in row.annotations]

        new_start, new_end, new_text = [], [], []

        # for each start-end index
        for s,e in zip(start,end):

            # extract the context until the start of the text
            tmp = context[:s]
            # compute the difference between the length of the context and of the cleaned context
            len_before = len(tmp)

            tmp_stripped = re.sub('\s', '', tmp)

            len_after = len(tmp_stripped)

            to_remove_first = len_before - len_after
            # define the new start index
            new_start.append(s-to_remove_first)

            # extract the context between the indices
            tmp = context[s:e]
            # compute the difference between the length of the context and of the cleaned context
            len_before = len(tmp)
            tmp_stripped = re.sub('\s', '', tmp)
            len_after = len(tmp_stripped)
            to_remove_after = len_before - len_after
            # # define the new end index
            new_end.append(e - (to_remove_first + to_remove_after))

            new_text.append(tmp_stripped)

        return new_start, new_end



    def make_bio_tagging(self, row : dict):
        """
          Tokenizes the input context and assignes a label to each token, solving the
          misalignment between labeled words and sub-tokens.

          Params:
            row : DataFrame row to tokenize
            tokenizer : Tokenizer to use
          Returns:
            context tokenized and associated labels in B-I-O format.
        """
        # compute new start/end indexes without considering white spaces
        char_wise_start, char_wise_end = self.adapt_indexes_without_spaces(row)

        # standard tokenization applied
        tokens_context = self.tokenizer.tokenize(row['data'], truncation=True, max_length=10000)

        # our result vector with a label for each token
        labels = ['O'] * len(tokens_context)

        # keep track of labels alreay assigned to token, distinguish between "B-" and "I-" labels
        labels_in_text = [annotation['label'][0] for annotation in row.annotations]
        mask_label_used = [False] * len(labels_in_text)

        # "pointer" (in the whole context without spaces) to first character of the current token
        actual_char_index = 0

        # most transformers' tokenizers add a special character to the first sub-token of a word
        # dummy tokenization to retrieve it
        init_special_char = self.tokenizer.tokenize('dummy')[0][0]
        if init_special_char == 'd':
          # bert models do not use special char for first sub-token, they use ## for all the other sub-tokens
          init_special_char = '##'

        for _, token in enumerate(tokens_context):
          # remove init character if present
          clean_tok = token.replace(init_special_char, "")

          for lbl_index , (start, end , label) in enumerate(zip(char_wise_start, char_wise_end, labels_in_text)):
            # check if the pointer is inside an entity
            if actual_char_index in range(start,end):
              if mask_label_used[lbl_index] == False:
                # first time we assign the label to a token
                labels[_] = "B-" + label
                # mark it as already assigned, next time will be "I-"
                mask_label_used[lbl_index] = True
              else:
                # the label has been already assigned to a token
                labels[_] = "I-" + label
              # once we have found the label we can skip the other checks
              break
          # update pointer
          actual_char_index += len(clean_tok)

        return tokens_context, labels

    
    def transform(self, df : pd.DataFrame, copy=False):
        tokens = []
        labels = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            tok, lab = self.make_bio_tagging(row)
            tokens.append(tok)
            labels.append(lab)

        if copy:
            return_df = df.copy()
            return_df['tokens'] = tokens
            return_df['labels'] = labels
            return return_df

        else:
            return_df = df
            return_df['tokens'] = tokens
            return_df['labels'] = labels
            return return_df
            




                
