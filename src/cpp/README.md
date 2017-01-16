
This software package contains a Barnes-Hut implementation of the t-SNE algorithm. The implementation is described in [this paper](http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf).


# Installation #

On Linux or OS X, compile the source using the following command:

```
g++ sptree.cpp tsne.cpp -o bh_tsne -O2
```

The executable will be called `bh_tsne`.

# Usage #

Copy the executable bh_tsne to your working directory (where you will use tsne.py). The python file tsne.py writes your data to a file called `.data.dat`, run the `bh_tsne` binary, and read the result file `.result.dat` that the binary produces.
