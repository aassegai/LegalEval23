from tqdm.auto import tqdm
import numpy as np
import torch


def embed(tok_texts, embedder, max_len=128, emb_size=512, batch_size=32):
    embedded = []
    for batch_start in tqdm(range(0, len(tok_texts['input_ids']), batch_size)):
        batch_output = embedder(input_ids=tok_texts['input_ids'][batch_start : batch_start + batch_size], 
                          attention_mask=tok_texts['attention_mask'][batch_start : batch_start + batch_size])
        # print(batch_output)
        hidden_state = batch_output['last_hidden_state'].detach().cpu().numpy()
        # print(hidden_state)
        embedded.append(hidden_state)
    return np.array(embedded).reshape(len(tok_texts['input_ids']), max_len, emb_size)