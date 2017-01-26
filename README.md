# tsne_visual (version 1.0)
A Python3 package for running, visualizing and producing animations of t-distributed stochastic Nearest-Neighbor Embedding (t-SNE) implemented in C++.
The code is a modified version of bhtsne taken from [Laurens van der Maaten repository](https://github.com/lvdmaaten/bhtsne). The package implements t-SNE as a Class following the [sklearn](http://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne) syntax. 

# Installing and running (in a few steps) :
- Clone or download this repository
- Compile C++ code: go to ```cpp/``` and run the command:
```
g++ sptree.cpp tsne.cpp -o bh_tsne -O2
```
- Copy the executable called ```bh_tsne``` to the directory you'll be running t-SNE (where your Python scripts are).
- Finally install the t-SNE package: from within the repository, use:
```
pip3 install .
```
# Example script for MNIST: 
For an example look at ```example/example.py```. This is an example of t-SNE applied to the MNIST data set (provided in ```example/MNIST/```).
The syntax used is very similar to [sklearn](http://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne) syntax. The C++ executable ```bh_tsne``` needs to be copied to the ```example/``` in order to run the ```example.py```.

# Requirements:
- Python3.x
- ffmpeg software (optional - for producing animations)
- Fairly common python packages:
  - sklearn
- For running example.py
  - seaborn 

# Why this package ?
During the course of a research project I ended up using t-SNE quite a bit for large datasets (N>20000). I wanted something 
easy to use (i.e. in written in python, and that gave me easy access to all of t-SNE parameters) but also very fast (i.e. with C/C++ speed). I also wanted to produce animations of the t-SNE as a function of the iterations. To achieve all this I ended combining codes from multiple sources and writing a bit of code myself. I thought this might be useful for other people too. 

# Some useful references:
- [t-SNE's original author website](https://lvdmaaten.github.io/tsne/)
- [sklearn t-SNE](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [Google's embedding projector](http://projector.tensorflow.org/)
- [How to use t-SNE effectively !](http://distill.pub/2016/misread-tsne/)

