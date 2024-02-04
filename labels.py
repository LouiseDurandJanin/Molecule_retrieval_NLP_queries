import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from k_means_constrained import KMeansConstrained


data = torch.load('/home/louisedurand-janin/Public/text_embeddings.pt')
#data = torch.load('/Users/elsaazoulay/Downloads/text_embeddings.pt')

#labels = MiniBatchKMeans(n_clusters=825, init='k-means++').fit_predict(data)
#np.save('/Users/elsaazoulay/Downloads/labels.npy', labels)

clf = KMeansConstrained(n_clusters=512, size_min=32, random_state=0)
labels = clf.fit_predict(data)

#np.save('/Users/elsaazoulay/Downloads/labels.npy', labels)
np.save('/home/louisedurand-janin/Public/labels.npy', labels)