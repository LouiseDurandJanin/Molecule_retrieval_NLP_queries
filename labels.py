import torch
import numpy as np
from k_means_constrained import KMeansConstrained


data = torch.load('./text_embeddings.pt')

clf = KMeansConstrained(n_clusters=512, size_min=32, random_state=0)
labels = clf.fit_predict(data)

np.save('./labels.npy', labels)