from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model import Model_baseline, ModelGAT, MLPModel, ModelGATv2
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
import time
import os
import argparse

batch_size = 32


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda available : ", torch.cuda.is_available())
parser = argparse.ArgumentParser()
parser.add_argument('--textmodel', type=str, default='distilbert-base-uncased', nargs='?',
                    help="text encoder model type from 'distilbert-base-uncased', 'allenai/scibert_scivocab_uncased'")



textmodel ='distilbert-base-uncased'




tokenizer = AutoTokenizer.from_pretrained(textmodel)

#Load GAT model trained 
model1 = ModelGAT(model_name=textmodel, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
model1.to(device)
save_path = os.path.join('./', 'modeltextbaseGAT'+'.pt')
checkpoint = torch.load(save_path)
model1.load_state_dict(checkpoint['model_state_dict'])

#Load Baseline model trained
model2 = Model_baseline(model_name=textmodel, num_node_features=300,ninp=768, nout=768, nhid=300, graph_hidden_channels=300) # nout = bert model hidden dim
model2.to(device)
save_path = os.path.join('./', 'modelbaseline'+'.pt')
checkpoint = torch.load(save_path)
model2.load_state_dict(checkpoint['model_state_dict'])

NUM_MODELS = 2
# Define a function to get the embeddings from a model
def get_embeddings(model, test_loader, test_text_loader):
    graph_embeddings = []
    text_embeddings = []
    model.eval()
    with torch.no_grad():
        graph_model = model.get_graph_encoder()
        text_model = model.get_text_encoder()
        for batch in test_loader:
            for output in graph_model(batch.to(device)):
                graph_embeddings.append(output.tolist())
        for batch in test_text_loader:
            for output in text_model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device)):
                text_embeddings.append(output.tolist())
    return text_embeddings, graph_embeddings

# Create data loaders for test text and graph datasets
test_cids_dataset = GraphDataset(root='./data/', gt=gt, split='test_cids')
test_text_dataset = TextDataset(file_path='./data/test_text.txt', tokenizer=tokenizer)

idx_to_cid = test_cids_dataset.get_idx_to_cid()

test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)

all_text_embeddings = []
all_graph_embeddings = []

# Get text and graph embeddings for model1
text_embeddings, graph_embeddings = get_embeddings(model1,  test_loader, test_text_loader)

all_text_embeddings.append(text_embeddings)
all_graph_embeddings.append(graph_embeddings)

# Get text and graph embeddings for model2
text_embeddings, graph_embeddings = get_embeddings(model2,  test_loader, test_text_loader)

all_text_embeddings.append(text_embeddings)
all_graph_embeddings.append(graph_embeddings)
# Ensemble method: Calculate cosine similarity between text and graph embeddings
ensemble_similarity = np.zeros((len(test_text_loader.dataset), len(test_loader.dataset)))

for i in range(NUM_MODELS):
    text_embeddings = all_text_embeddings[i]
    graph_embeddings = all_graph_embeddings[i]

    for j in range(len(test_text_loader.dataset)):
        for k in range(len(test_loader.dataset)):
            similarity = cosine_similarity([text_embeddings[j]], [graph_embeddings[k]])
            ensemble_similarity[j, k] += similarity[0, 0]

# Normalize the ensemble similarity scores (optional)
ensemble_similarity /= NUM_MODELS

# Create a DataFrame for the submission
solution = pd.DataFrame(ensemble_similarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv('ensemble_submission.csv', index=False)
