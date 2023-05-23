from spacy import displacy
import pandas as pd
from tqdm.auto import tqdm


def make_annotation(data,
                   predictions):

    new_data = data

    idx = 0

    for doc in tqdm(new_data):
        for annotation in doc['annotations']:
            for sentence in annotation['result']:
                sentence['value']['labels'][0] = predictions[idx]
                sentence += 1

    return new_data


def show_text_segmentation(doc, annotation):
    display_sentences = []
    for item in annotation:
        new_item = {'start': item['start'],
                    'end': item['end'],
                    'label': item['label'][0]}
        display_sentences.append(new_item)

    display_dict = {'text': doc,
                    'ents': annotation}
    displacy.render(display_dict, manual=True, style='ent')