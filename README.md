# Altegrad_data_challenge

## Repository linked to the Kaggle Data Challenge on Molecule Retrieval using Natural Language Queries 
## Altegrad class, MVA, ENS Paris-Saclay

```
├── dataloader.py       <- Script to load the graphs and NLP queries
│
├── Model.py           <- Script that instantiate the models used
│
├── main.py            <- Train the model putted as argument
│     
├── requirements.txt   <- Required libraries and dependencies. 
│
├── predictions.py     <- Script to make predictions using the model added in argument
│
├── labels.py          <- Scrip to get the batch's labels associated with each embedding
│   
├── main_negativemining.py <- Script to train a model using negative mining 


│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models, to use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
```
