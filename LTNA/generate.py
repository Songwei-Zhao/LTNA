import transformers
import torch
from enhance import *
import pandas as pd
# from sentence_transformers import SentenceTransformer
# from langchain.embeddings import LlamaCppEmbeddings
import random
import numpy as np
from tqdm import tqdm
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"

model_id = "Meta-Llama-3-8B"


model = transformers.AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
# model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_obj = torch.load("datasets/cora_random.pt")
texts = data_obj.raw_texts
num_nodes = len(data_obj.raw_texts)
adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
for i, j in zip(data_obj.edge_index[0], data_obj.edge_index[1]):
     adj[i, j] = 1

embeddings = []
for i, text in enumerate(tqdm(texts)):
# for node in sampled_nodes:

    # label = data_obj.category_names[i]
    neighbors = find_neighbors(adj, i)
    # print(neighbors)
    # neighbors_info = [data_obj.raw_texts[neighbor] for neighbor in neighbors]
    # # neighbors_label = [data_obj.category_names[neighbor] for neighbor in neighbors]
    neighbors_info = [texts[neighbor] for neighbor in neighbors]

    sampled_neighbors_info = sample_neighbors(neighbors_info, max_samples = 10)

    prompt = create_aggregator_llama_prompt(text, sampled_neighbors_info)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    emb = outputs.last_hidden_state.mean(dim=1).squeeze()  # 
    

    # hidden_states = outputs.last_hidden_state

    # sentence_embedding = hidden_states[:, 0, :].squeeze()

    if i == 0:
        embedding_dim = emb.size(0)  
        all_embeddings = torch.zeros((data_obj.num_nodes, embedding_dim), dtype=torch.float32)

    # # all_embeddings.append(sentence_embedding)
    # # all_embeddings.append(sentence_embedding.cpu().to(torch.float32).numpy())
    all_embeddings[i, :] = emb

    # print(all_embeddings)
# return torch.stack(embeddings)
# print(all_embeddings)

with open("cora.pkl", "wb") as f:
    pickle.dump(all_embeddings, f)

print("All embeddings have been saved to cora.pkl.")

