from tqdm.auto import tqdm
import numpy as np
import torch


def embed(tok_texts, embedder, max_len=128, emb_size=512, batch_size=32, device='cpu'):
    embedded = []
    embedder.to(device=device)
    for batch_start in tqdm(range(0, len(tok_texts['input_ids']), batch_size)):
        batch_output = embedder(input_ids=tok_texts['input_ids'][batch_start : batch_start + batch_size].to(device=device), 
                          attention_mask=tok_texts['attention_mask'][batch_start : batch_start + batch_size].to(device=device))
        # print(batch_output)
        hidden_state = batch_output['last_hidden_state'].detach().cpu().numpy()
        # print(hidden_state)
        embedded.extend(hidden_state)
    return np.array(embedded)