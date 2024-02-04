from Model import Model
import numpy as np
from transformers import AutoTokenizer
from dataloader import GraphTextDataset, GraphDataset, TextDataset
import torch
from torch_geometric.data import DataLoader
from sklearn.cluster import MiniBatchKMeans, KMeans
from tqdm import tqdm

#escription = pd.read_csv('./data/train.tsv', sep='\t', header=None)
#description = description.set_index(0).to_dict()

#cids = list(description[1].keys())

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(model_name=model_name, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300) # nout = bert model hidden dim
model.to(device)
save_path = '/home/elsaazoulay/Public/basemodel.pt'
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])

text_encoder = model.get_text_encoder()

batch_size = 32

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

text_embeddings = []

model.eval()

for batch in tqdm(train_loader):
    for output in text_encoder(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device)):
        text_embeddings.append(output.tolist())

emb = torch.tensor(text_embeddings)
torch.save(emb, 'text_embeddings_base.pt')
