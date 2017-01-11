import numpy as np
import os
from tsne import TSNE
from sklearn import datasets

print(os.getcwd())
np.random.seed(0)
tsne=TSNE()
X,y=datasets.make_blobs(n_samples=10000,n_features=20,centers=20,random_state=0)
tsne.fit(X)
#os.system("./dev_bhtsne-master/bh_tsne -h")
