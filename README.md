# Molecule Retrieval with Natural Language Queries

Repository linked to the Kaggle Data Challenge on Molecule Retrieval using Natural Language Queries 

**_Altegrad class, MVA, ENS Paris-Saclay_**

**Abstract**

*This project explores machine learning and artificial intelligence techniques to address the challenge of retrieving molecules (represented as graphs) based on natural language queries. This task requires integrating two fundamentally different modalities: the structured semantic information encoded in text and the chemical properties represented by molecular graphs. The objective is to map these modalities into a shared latent space where semantically similar text-molecule pairs are aligned, using contrastive learning.*

*Our approach involves co-training a text encoder and a molecule encoder, where the text encoder processes the natural language queries and the molecule encoder handles graph-structured data. Various strategies were tested, including baseline models, enhancements to the graph encoder with attention mechanisms (GATs and GATv2), exploration of domain-specific text encoders such as SciBERT, and the incorporation of hard negative mining to improve representation learning. Additionally, ensemble methods were investigated to better integrate diverse learned patterns from different models.*

*This work demonstrates the potential of cross-modal learning frameworks for bridging textual and molecular representations and highlights the challenges and opportunities in multi-modal retrieval tasks.*

**Description**

* dataloader.py : Load the graphs and NLP queries
* model.py : Instantiate the models used
* main.py : Train the model
* requirements.txt : Required libraries and dependencies
* predictions.py : Make predictions using the model added in argument
* get_text_embeddings.py : Extract and save final text embeddings from our trained GAT model
* labels.py : Get the batch's labels associated with each embedding
* main_negativemining.py : Train our GAT model using negative mining
* ensemble_method.py : Ensemble method that computes the mean over cosine similarity of different models

