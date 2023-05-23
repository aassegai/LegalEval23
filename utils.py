from spacy import displacy
import pandas as pd
from tqdm.auto import tqdm
from IPython.core.display import display, HTML


def make_annotation(data,
                   predictions):

    new_data = data

    idx = 0

    for _, doc in new_data.iterrows():
        for sentence in doc.annotations:
            sentence['label'][0] = predictions[idx]
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
  
    html = displacy.render(display_dict, manual=True, style='ent')
    display(HTML(html))