'''
Created on Dec 23, 2016

@author: Alexandre Day


Purpose:
    This is a python wrapper/package for performing t-SNE embeddings.
    It defines a tSNE class which mimics the syntax of sklearn package with similar methods 
    It adds new methods to produce animations of the t-SNE embedding
    and is quite fast since the underlying code is written in C++. 
    See README.md and license for authors of the C++ code. The original 
    bh_tsne C++ code was modified to accomodate more options.
'''

from . import utilities as ut
import numpy as np

class TSNE:

    """t-distributed Stochastic Neighbor Embedding.

    t-SNE is a tool to visualize high-dimensional data. It converts
    similarities between data points to joint probabilities and tries
    to minimize the Kullback-Leibler divergence between the joint
    probabilities of the low-dimensional embedding and the
    high-dimensional data. t-SNE has a cost function that is not convex,
    i.e. with different initializations we can get different results.

    It is highly recommended to use another dimensionality reduction
    method (e.g. PCA for dense data or TruncatedSVD for sparse data)
    to reduce the number of dimensions to a reasonable amount (e.g. 50)
    if the number of features is very high. This will suppress some
    noise and speed up the computation of pairwise distances between
    samples. For more tips see Laurens van der Maaten's FAQ.
    
    Conventionally, data is saved in data.dat and tSNE results are 
    saved in result.dat
    
    Ref.
        [1] http://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne
        [2] https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding
        [3] https://lvdmaaten.github.io/tsne/  
    
    Parameters
    ----------
    n_components : int, optional (default: 2)
        Dimension of the embedded space.

    perplexity : float, optional (default: 30)
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. The choice is not extremely critical since t-SNE
        is quite insensitive to this parameter.

    early_exaggeration : float, optional (default: 4.0)
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them. For
        larger values, the space between natural clusters will be larger
        in the embedded space. Again, the choice of this parameter is not
        very critical. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high.

    learning_rate : float, optional (default: 1000)
        The learning rate can be a critical parameter. It should be
        between 100 and 1000. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high. If the cost function gets stuck in a bad local
        minimum increasing the learning rate helps sometimes.

    n_iter : int, optional (default: 1000)
        Maximum number of iterations for the optimization. Should be at
        least 200.

    n_iter_without_progress : int, optional (default: 30)
        Only used if method='exact'
        Maximum number of iterations without progress before we abort the
        optimization. If method='barnes_hut' this parameter is fixed to
        a value of 30 and cannot be changed.

        .. versionadded:: 0.17
           parameter *n_iter_without_progress* to control stopping criteria.

    min_grad_norm : float, optional (default: 1e-7)
        Only used if method='exact'
        If the gradient norm is below this threshold, the optimization will
        be aborted. If method='barnes_hut' this parameter is fixed to a value
        of 1e-3 and cannot be changed.

    metric : string or callable, optional
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them. The default is "euclidean" which is
        interpreted as squared euclidean distance.

    init : string or numpy array, optional (default: "random")
        Initialization of embedding. Possible options are 'random', 'pca',
        and a numpy array of shape (n_samples, n_components).
        PCA initialization cannot be used with precomputed distances and is
        usually more globally stable than random initialization.

    verbose : int, optional (default: 0)
        Verbosity level.

    random_state : int or RandomState instance or None (default)
        Pseudo Random Number generator seed control. If None, use the
        numpy.random singleton. Note that different initializations
        might result in different local minima of the cost function.

    method : string (default: 'barnes_hut')
        By default the gradient calculation algorithm uses Barnes-Hut
        approximation running in O(NlogN) time. method='exact'
        will run on the slower, but exact, algorithm in O(N^2) time. The
        exact algorithm should be used when nearest-neighbor errors need
        to be better than 3%. However, the exact method cannot scale to
        millions of examples.

        .. versionadded:: 0.17
           Approximate optimization *method* via the Barnes-Hut.

    angle : float (default: 0.5)
        Only used if method='barnes_hut'
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.
        If angle < 1e-3, method='exact' will be used.
        
    animate : bool (default: False)
        Produce a .mp4 clip of the t-SNE iterations. This is useful to gain intuition
        about t-SNE and visualizing convergence (see http://distill.pub/2016/misread-tsne/ for a useful tutorial)  
        
    file_name: string (default 'auto')
        Name of file where embedding to write the embedded data. If 'auto', will
        automatically generate a file name (with parameters indicated in the file name)
    
    n_iter_lying: int (default 200)
        Number of iteration that P-values are exaggerated. After 'n_iter_lying' iterations,
        P-values are set to their real value
        
    n_iter_momentum_switch: int (default 200)
        Number of iteration that momentum is set to 0.5 (updates are equally averaged)
        After 'n_iter_momentum_switch' iterations, gradient descent updates are more reliable
        and thus we increase momentum to 0.8
    
    """ 
       
        
    def __init__(self, n_components=2, perplexity=30.0,
                 early_exaggeration=4.0, learning_rate=100.0, n_iter=1000,
                 n_iter_without_progress=30, min_grad_norm=1e-7,#metric="euclidean", 
                 init="random", PCA_n_components=None,verbose=0,
                 random_state=None, method='barnes_hut', angle=0.5,
                 animate=False,
                 n_iter_lying=200,n_iter_momentum_switch=200
                 ):
        if not (init in ["pca", "random"]):
            msg = "'init' must be 'pca', 'random'"
            raise ValueError(msg)
        
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.n_iter_lying =n_iter_lying
        self.n_iter_momentum_switch =n_iter_momentum_switch
        
        #self.metric = metric
        self.init = init
        self.PCA_n_components=PCA_n_components
        self.verbose = verbose
        
        if random_state is None:
            self.random_state = np.random.randint(0,2147483647)
        else:
            self.random_state = random_state
                        
        self.method = method
        self.angle = angle
        self.embedding_ = None
        
        self.file_name="result.dat"
        
    def fit(self, X):
        import os
        """
        Purpose:
            Fit X into the embedded space using the C++ executable
        
        Parameter:
            X : array, shape (n_samples, n_features)
        """
        
        assert(len(X.shape)==2), "X must be a 2D array"
        
        ut.data_to_binary(X,delimiter="\t")
        
        parameters=[self.n_components,self.perplexity,
                    self.early_exaggeration,self.learning_rate,
                    self.angle,self.random_state,
                    self.n_iter,self.n_iter_without_progress,
                    self.min_grad_norm,self.n_iter_lying,
                    self.n_iter_momentum_switch,self.verbose,
                    X.shape[0],X.shape[1],os.getcwd()+"/"
                    ]
                    
        ut.run_tsne_command_line(parameters)
        
        self.embedding_ = np.fromfile(self.file_name).reshape(-1,self.n_components)
        
    def fit_transform(self,X):
        """
        Purpose:
            Fit X into the embedded space using the C++ executable
        
        Parameter:
            X : array, shape (n_samples, n_features)
        """
        self.fit(X)
        
        return self.embedding_
    
