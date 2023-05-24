from spacy import displacy
import pandas as pd
from tqdm.auto import tqdm
from IPython.core.display import display, HTML
from matplotlib import pyplot as plt
import numpy as np

def make_annotation(data,
                   predictions, id2label):

    new_data = data.copy()

    idx = 0

    for _, doc in new_data.iterrows():
        for sentence in doc.annotations:
            sentence['label'][0] = id2label[predictions[idx]]
            idx += 1

    return new_data


def show_text_segmentation(doc, annotation):
    display_sentences = []
    for item in annotation:
        new_item = {'start': item['start'],
                    'end': item['end'],
                    'label': item['label'][0]}
        display_sentences.append(new_item)

    display_dict = {'text': doc,
                    'ents': display_sentences}

    colors = {'PREAMBLE': '#FF0000',
              'FAC': '#00FF00',
              'RLC': '#35B8D0',
              'ISSUE': '#FF00FF',
              'ARG_PETITIONER': '#FFFF00',
              'ARG_RESPONDENT': '#00FFFF',
              'ANALYSIS': '#5EAF48',
              'STA': '#35B8D0',
              'PRE_RELIED': '#008000',
              'PRE_NOT_RELIED': '#FFC0CB',
              'RATIO': '#800000',
              'RPC': '#FF7F00',
              'NONE': '000010'}

    options = {'colors': colors}
  
    html = displacy.render(display_dict, 
                            manual=True, style='ent', options=options)
    display(HTML(html))


def aggregate_label_stats(df: pd.DataFrame):
    temp_df = df.copy()

    stats_df = temp_df.groupby(['label']).count()
    temp_df['sentence_len'] = temp_df['sentence'].apply(lambda x: len(x.split()))
    stats_df['mean_sentence_len'] = temp_df.groupby(['label']).mean(['sentence_len']).sentence_len
    stats_df['max_sentence_len'] = temp_df.groupby(['label']).max(['sentence_len']).sentence_len

    temp_df['context_len'] = temp_df['context'].apply(lambda x: len(x.split()))
    stats_df['mean_context_len'] = temp_df.groupby(['label']).mean(['context_len']).context_len
    
    stats_df.drop(columns=['text', 'context', 'sentence'], inplace=True)
    stats_df.rename(columns={'doc_id': 'count'}, inplace=True)
    stats_df.reset_index(inplace=True)


    return stats_df


def display_label_counts(df: pd.DataFrame):

    fig, ax = plt.subplots(1, 1, figsize=(22, 5))
    labels_list = df['label'].to_list()
    for label in sorted(set(labels_list)):
        plt.hist([label_item for label_item in labels_list if label_item == label], 
        bins=np.arange(14)-0.5, rwidth=0.8)
    plt.show()