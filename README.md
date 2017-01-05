# tSNE_visual
A Python3 wrapper for running,visualizing and producing animations of t-distributed stochastic nearest-neighbor embedding implemented in C++.
The code is a modified version of bhtsne taken from Laurens van der Maaten repository. The wrapper uses a similar syntax
to sklearn, where tsne is defined as a class.

# Compiling C++:
Go to /src/cpp and run:
```
g++ sptree.cpp tsne.cpp -o bh_tsne -O2
```

# Running from command line:
From /src run:
```
$python tsne example.txt 40 4.0 1000 0.5 out_data_file.txt
```

For details on the meaning of those parameters, run:
```
$python tsne example.txt 40 4.0 1000 0.5 out_data_file.txt
```

# Requirements:
- Python3.x
- ffmpeg software  
