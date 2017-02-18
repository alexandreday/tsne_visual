import numpy as np
from tsne_visual import TSNE
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
n_iter=1000

tsne=TSNE(n_iter=n_iter) ## Otherwise use something like TSNE(perplexity=50, n_iter=2000, angle=0.3)
start=time.time() 
xtsne=tsne.fit_transform(xpca)
dt=time.time()-start

print("Finally let's plot the results")

# Some colors !
palette=["#1CE6FF", "#FF34FF", "#FF4A46","#008941", "#006FA6", "#A30059", '#008080','#FFA500', '#3399FF','#800080']

for i in range(10):
    pos=(z==i).flatten()
    plt.scatter(xtsne[pos,0], xtsne[pos,1],
        c=palette[i],
        label=str(i),
        edgecolor='black'
    )

plt.title('Applying t-SNE, # of samples = 10000 \n with %i iterations, run time=%.2f s'%(n_iter,dt))
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(loc='best')
plt.tight_layout()
print("Saving plot")
plt.savefig('result.png')
print('Showing plot, voila !')
plt.show()

