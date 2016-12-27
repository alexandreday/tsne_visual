# tSNE_visual
Python/C++ simple tools for running and visualizing t-SNE to produce low dimensional embeddings
The t-SNE is implemented in C++ and is a modified code taken from Laurens van der Maaten github reposity: https://github.com/lvdmaaten/bhtsne/
A python wrapper is used to call the C++ executable and make it user friendly. 
The python wrapper uses the sklearn (http://scikit-learn.org/stable/) syntax structure.
The user can choose the t-SNE hyperparameters and also produce a movie of the iteration steps during the
gradient descent. This requires the ffmpeg software.
 
 
