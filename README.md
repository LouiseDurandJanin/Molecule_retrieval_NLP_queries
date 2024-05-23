# Altegrad_data_challenge

## Repository linked to the Kaggle Data Challenge on Molecule Retrieval using Natural Language Queries 
## Altegrad class, MVA, ENS Paris-Saclay

```
├── dataloader.py          <- Load the graphs and NLP queries
│
├── Model.py               <- Instantiate the models used
│
├── main.py                <- Train the model
│     
├── requirements.txt       <- Required libraries and dependencies. 
│
├── predictions.py         <- Make predictions using
│                             the model added in argument
│
├── get_text_embeddings.py <- Extract and save final text 
│                             embeddings from our trained GAT model
│
├── labels.py              <- Get the batch's labels 
│                             associated with each embedding
│   
├── main_negativemining.py <- Train our GAT model using 
│                             negative mining
│ 
├── ensemble_method.py     <- Ensemble method that computes the mean over cosine similarity of different models

```
