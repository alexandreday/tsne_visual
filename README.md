# tsne_visual (version 0.2) - for OSX and Linux
A Python3 package for running, visualizing and producing animations of t-distributed stochastic Nearest-Neighbor Embedding (t-SNE) implemented in C++.
The code is a modified version of bhtsne taken from [Laurens van der Maaten repository](https://github.com/lvdmaaten/bhtsne). The package implements t-SNE as a Class following the [sklearn](http://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne) syntax.

# Installing and running (in a few steps) :
- Clone or download this repository
- Open file
- Compile C++ code and install the package with the following commands:
```
g++ cpp/sptree.cpp cpp/tsne.cpp -o tsne_visual/bh_tsne -O2
pip3 install .
```
That's it, you're good to go ! You can now import ```tsne_visual``` from anywhere. See the following example
for a quick start.
# Example script for MNIST: 
For an example look at ```example/example.py```. This is an example of t-SNE applied to the MNIST data set (provided in ```example/MNIST/```).
The syntax used is very similar to [sklearn](http://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne) syntax.
It should a produce a figure similar to this: ![alt tag](https://github.com/alexandreday/tsne_visual/blob/master/gallery/MNIST-1.png)

# Requirements:
- g++ compiler (for the C++ code)
- Python3.x
- ffmpeg software (optional - for producing animations)
- [scikit-learn package](http://scikit-learn.org/stable/install.html)

# Why this package ?
During the course of a research project I ended up using t-SNE quite a bit for large datasets (N>20000). I wanted something 
easy to use (i.e. in written in python, and that gave me easy access to all of t-SNE parameters) but also very fast (i.e. with C/C++ speed). I also wanted to produce animations of the t-SNE as a function of the iterations. To achieve all this I ended combining codes from multiple sources and writing a bit of code myself. I thought this might be useful for other people too. 

# Some useful references:
- [t-SNE's original author website](https://lvdmaaten.github.io/tsne/)
- [sklearn t-SNE](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [Google's embedding projector](http://projector.tensorflow.org/)
- [How to use t-SNE effectively !](http://distill.pub/2016/misread-tsne/)

# Uninstalling !
```
pip3 uninstall tsne_visual
```
