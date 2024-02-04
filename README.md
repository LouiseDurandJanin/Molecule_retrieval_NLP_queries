# Altegrad_data_challenge

## Repository linked to the Kaggle Data Challenge on Molecule Retrieval using Natural Language Queries 
## Altegrad class, MVA, ENS Paris-Saclay

```
├── dataloader.py          <- Script to load the graphs and NLP queries
│
├── Model.py               <- Script that instantiate the models used
│
├── main.py                <- Train the model putted as argument
│     
├── requirements.txt       <- Required libraries and dependencies. 
│
├── predictions.py         <- Script to make predictions using
│                             the model added in argument
│
├── get_text_embeddings.py <- Script to extract and save final text 
│                             embeddings from our trained GAT model
│
├── labels.py              <- Scrip to get the batch's labels 
│                             associated with each embedding
│   
├── main_negativemining.py <- Script to train our GAT model using 
│                             negative mining
│ 
├── ensemble_method.py     <- Ensemble method that computes the mean over cosine similarity of different models

```
