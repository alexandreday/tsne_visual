import numpy as np
from tsne_visual.tsne import TSNE
from read_MNIST import load_mnist
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import time

print("--> Running t-SNE on MNIST (with n=10000), then plotting the result")

np.random.seed(0) # Always same seed here <--
x,y=load_mnist(path="MNIST/")

x=np.array(x[:10000],dtype=np.float)
z=y[:10000]

print("--> First doing some PCA (n_components=40) to clear irrelevant dimensions")
pca=PCA(n_components=40)
xpca=pca.fit_transform(x)

## Running t-SNE with default parameters !
tsne=TSNE(n_iter=300) ## Otherwise use something like TSNE(perplexity=50, n_iter=2000, angle=0.3)
start=time.time() 
xtsne=tsne.fit_transform(xpca)
dt=time.time()-start

print("Finally let's plot the results")
palette=np.array(sns.color_palette('hls',10))

for i in range(10):
    pos=(z==i).flatten()
    plt.scatter(xtsne[pos,0],xtsne[pos,1],c=palette[i],label=str(i))

plt.title('t-SNE with (# of sample = 10000) and run for %i iterations\n run time=%.2f s'%(n_iter,dt))
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(loc='best')
plt.tight_layout()
plt.grid(True)
print('Voila !')
plt.show()

