import numpy as np
import seaborn as sns
from tsne.tsne import TSNE
from read_MNIST import load_mnist
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import time

np.set_printoptions(suppress=True)

print("--> Running t-SNE on MNIST (with n=10000), then plotting the result")

np.random.seed(0)
x,y=load_mnist(path="MNIST/")

x=np.array(x[:10000],dtype=np.float)
z=y[:10000]

print("--> First doing some PCA (n_components=40) to clear irrelevant dimensions")
pca=PCA(n_components=40)
xpca=pca.fit_transform(x)

n_iter=1000
tsne=TSNE(n_components=2,early_exaggeration=12.0, learning_rate=200.0, n_iter=n_iter, 
                 n_iter_lying=250,n_iter_momentum_switch=250
        )
        
print("--> Now let's run the t-SNE for %i iterations !\n\n"%n_iter)
start=time.time() 
xtsne=tsne.fit_transform(xpca)
dt=time.time()-start

print("Finally let's plot the results")
palette=np.array(sns.color_palette('hls',10))

for i in range(10):
    pos=(z==i).flatten()
    plt.scatter(xtsne[pos,0],xtsne[pos,1],c=palette[i],label=str(i))

plt.title('t-SNE with (Nsample=10000) and run for %i iterations\n Run time=%.2f s'%(n_iter,dt))
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(loc='best')
print('Voila !')
plt.show()

