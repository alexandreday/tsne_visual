import numpy as np
import sys
sys.path.append("..")
import seaborn as sns
from tsne import TSNE
from read_MNIST import load_mnist
from sklearn.decomposition import PCA
from sklearn import preprocessing as prep
from matplotlib import pyplot as plt

np.set_printoptions(suppress=True)

print("Running t-SNE on MNIST (with n=10000), then plotting the result")

np.random.seed(0)
x,y=load_mnist(path="MNIST/")

x=np.array(x[:10000],dtype=np.float)
z=y[:10000]

print("First scaling results and doing some PCA (n_components=40)")
x_scale=prep.scale(x)
pca=PCA(n_components=40)
xpca=pca.fit_transform(x_scale)

tsne=TSNE(n_components=2,early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, 
                 n_iter_lying=250,n_iter_momentum_switch=250
        )
 
print("Now let's run the t-SNE !\n\n")
xtsne=tsne.fit_transform(xpca)

print("Finally let's plot the results")
palette=np.array(sns.color_palette('hls',10))

for i in range(10):
    pos=(z==i).flatten()
    plt.scatter(xtsne[pos,0],xtsne[pos,1],c=palette[i],label=str(i))

plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(loc='best')
print('Voila !')
plt.show()

