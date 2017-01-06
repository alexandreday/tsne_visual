import numpy as np
from tsne import TSNE


X=np.loadtxt("example.txt",dtype=np.float32,delimiter="\t")
print("Successfully read file... Contains %i samples and %i dimensions"%X.shape)
f=TSNE(verbose=0)
f.fit(X)
