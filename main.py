from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model import Model_baseline, ModelGAT, ModelGATv2
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
import time
import os
import pandas as pd
import argparse

print("cuda available : ", torch.cuda.is_available())


# Define contrastive loss 
CE = torch.nn.CrossEntropyLoss()
def contrastive_loss(v1, v2):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  labels = torch.arange(logits.shape[0], device=v1.device)
  return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)


# Run the model you want passing it as an argument 
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Base', nargs='?',
                    help="model type from 'Base', 'GATbase', 'GATv2', 'GATScibert'")
args = parser.parse_args() 
MODEL = args.model

nb_epochs = 5
batch_size = 16
learning_rate = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Define and instantiate each model

if MODEL =='Base':
    # Baseline model provided

    print("Baseline Model")
    model_name = "baseline"
    text_model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
    train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)
    model = Model_baseline(model_name=text_model_name, num_node_features=300,ninp=768, nout=768, nhid=300, graph_hidden_channels=300) # nout = bert model hidden dim
    model.to(device)

elif MODEL =='GATbase':
    # DistillBert Text Encoder and GATConv used in Graph Decoder

    print("GAT Base Model")
    model_name = "gatbase"
    text_model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
    train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)
    model = ModelGAT(model_name=text_model_name, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
    model.to(device)
    ### Uncomment to load previously saved model

    #save_path = os.path.join('./', f'{MODEL}model.pt')
    #checkpoint = torch.load(save_path)
    #model.load_state_dict(checkpoint['model_state_dict'])

elif MODEL == "GATv2":
    # DistillBert Text Encoder and GATv2Conv used in Graph Decoder

    print("GATv2 Model")
    model_name = "gatv2base"
    text_model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
    train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)
    model = ModelGATv2(text_model_name, num_node_features=300,ninp=768, nout=768, nhid=300, graph_hidden_channels=300) 
    model.to(device)

elif MODEL =='GATScibert':
    # SciBert Text Encoder and GATConv used in Graph Decoder

    print("GAT Scibert Model")
    model_name = "gatscibert"
    text_model_name = 'allenai/scibert_scivocab_uncased'
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
    train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)
    model = ModelGAT(model_name=text_model_name, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300) # nout = bert model hidden dim
    model.to(device)
    ### Uncomment to load previously saved model

    #save_path = os.path.join('./', f'{MODEL}model.pt')
    #checkpoint = torch.load(save_path)
    #model.load_state_dict(checkpoint['model_state_dict'])





val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Defining optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                betas=(0.9, 0.999),
                                weight_decay=0.01)

epoch = 0
loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 50
best_validation_loss = 1000000



for i in range(nb_epochs):
    print('-----EPOCH{}-----'.format(i+1))
    # Train model 

    model.train()
    for batch in train_loader:
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        
        x_graph, x_text = model(graph_batch.to(device), 
                                input_ids.to(device), 
                                attention_mask.to(device))
        current_loss = contrastive_loss(x_graph, x_text)   
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        loss += current_loss.item()
        
        count_iter += 1
        if count_iter % printEvery == 0:
            time2 = time.time()
            print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                        time2 - time1, loss/printEvery))
            losses.append(loss)
            loss = 0 
    # Evaluate model on valdation set
            
    model.eval()       
    val_loss = 0        
    for batch in val_loader:
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        x_graph, x_text = model(graph_batch.to(device), 
                                input_ids.to(device), 
                                attention_mask.to(device))
        current_loss = contrastive_loss(x_graph, x_text)   
        val_loss += current_loss.item()
    best_validation_loss = min(best_validation_loss, val_loss)
    print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)) )
    if best_validation_loss==val_loss:
        print('validation loss improoved saving checkpoint...')
        save_path = os.path.join('./', f'{MODEL}model.pt')
        torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_accuracy': val_loss,
        'loss': loss,
        }, save_path)
        print('checkpoint saved to: {}'.format(save_path))


print('loading best model...')
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

graph_model = model.get_graph_encoder()
text_model = model.get_text_encoder()

test_cids_dataset = GraphDataset(root='./data/', gt=gt, split='test_cids')
test_text_dataset = TextDataset(file_path='./data/test_text.txt', tokenizer=tokenizer)

idx_to_cid = test_cids_dataset.get_idx_to_cid()

test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

graph_embeddings = []
for batch in test_loader:
    for output in graph_model(batch.to(device)):
        graph_embeddings.append(output.tolist())

test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
text_embeddings = []
for batch in test_text_loader:
    for output in text_model(batch['input_ids'].to(device), 
                             attention_mask=batch['attention_mask'].to(device)):
        text_embeddings.append(output.tolist())

# Make predictions

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(text_embeddings, graph_embeddings)

solution = pd.DataFrame(similarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv('submission.csv', index=False)