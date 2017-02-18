all:
	g++ cpp/sptree.cpp cpp/tsne.cpp -o tsne_visual/bh_tsne -O2
	pip333 install .
