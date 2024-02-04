from torch import nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv, GATv2Conv, GATConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel


# Graph Encoder from Baseline
class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x

# Text Encoder from Baseline
class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        return encoded_text.last_hidden_state[:,0,:]
    

# Baseline model   
class Model_baseline(nn.Module):
    def __init__(self, model_name, num_node_features,nout, nhid, graph_hidden_channels):
        super(Model_baseline, self).__init__()
        self.graph_encoder = GraphEncoder(num_node_features, nout, nhid, graph_hidden_channels)
        self.text_encoder = TextEncoder(model_name)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
    
# Graph Encoder using GATConv
class GraphEncoderGAT(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, dropout=0.2):
        super(GraphEncoderGAT, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.gat1 = GATConv(num_node_features, graph_hidden_channels, heads=10, dropout=dropout)
        self.gat2 = GATConv(graph_hidden_channels * 10, graph_hidden_channels, dropout=dropout)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x1 = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x1, weight = self.gat1(x1, edge_index,return_attention_weights=True)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)

        x1, weight2 = self.gat2(x1, edge_index, return_attention_weights=True)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = global_mean_pool(x1, batch)
        x1 = self.mol_hidden1(x1).relu()
        x1 = self.mol_hidden2(x1)

        return x1

# Model using GATConv
class ModelGAT(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels):
        super(ModelGAT, self).__init__()
        self.graph_encoder = GraphEncoderGAT(num_node_features, nout, nhid, graph_hidden_channels)
        self.text_encoder = TextEncoder(model_name)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder

#Graph Encoder using GATv2Conv
class GraphEncoderGATv2(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GraphEncoderGATv2, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GATv2Conv(num_node_features, graph_hidden_channels, heads=5)
        self.conv2 = GATv2Conv(graph_hidden_channels, graph_hidden_channels, heads=5)
        self.conv3 = GATv2Conv(graph_hidden_channels, graph_hidden_channels, heads=5)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x

# Model using GATv2Conv
class ModelGATv2(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels):
        super(ModelGATv2, self).__init__()
        self.graph_encoder = GraphEncoderGATv2(num_node_features, nout, nhid, graph_hidden_channels)
        self.text_encoder = TextEncoder(model_name)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder



    


