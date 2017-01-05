# tSNE_visual (Under Development)
A Python3 wrapper for running, visualizing and producing animations of t-distributed stochastic Nearest-Neighbor Embedding (t-SNE) implemented in C++.
The code is a modified version of bhtsne taken from [Laurens van der Maaten repository](https://github.com/lvdmaaten/bhtsne). The wrapper uses a similar syntax
to [sklearn](http://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne), where t-SNE is defined as a class.

# Compiling C++ code:
Go to src/cpp and run:
```
g++ sptree.cpp tsne.cpp -o bh_tsne -O2
```

# Running from command line:
From src/ run:
```
$python tsne example.txt 40 4.0 1000 0.5 out_data_file.txt
```

For details on the meaning of those parameters, run:
```
$python tsne -h
```

# Requirements:
- Python3.x
- ffmpeg software  
- Python packages:
  - sklearn
  - pandas

# Why this wrapper ?
During the course of a research project I ended up using t-SNE quite a bit for large datasets (N>20000). I wanted something 
easy to use (i.e. in written in python !) but also very fast (i.e. with C/C++ speed). I also wanted to produce animations 
of the t-SNE. To achieve all this I ended combining codes from multiple sources and writing a bit of code myself. I thought this
might be useful for other people too so this is why I uploaded the code as a github repo.

