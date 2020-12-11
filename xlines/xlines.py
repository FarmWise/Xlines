
from . import utils
from .klines import KLines
import numpy as np


class XLines(object):
    
    def __init__(self, candidates, alpha0=0., init_alpha="one", max_iter=10, tol=0.001, 
                 clustering_n_init=3, clustering_init="estimate", metric="silhouette", 
                 verbose=0):
        """
        Initialize a XLines object

        Parameters
        ---
        candidates : array-like, list of candidate k for which we will try to fit a KLines
        alpha0 : float, initial value for alpha if init_alpha is None or "update"
        init_alpha : {None, "one", "all", "update"}, if not None test a few alpha0 and 
            initialize the iterations with the best one.
            If "one", this step is done only for the first candidate and later 
            ones use the resulting orientation as alpha0.
            If "update", first candidate uses alpha0 and later ones use the 
            updated alpha0.
        max_iter : int or array-like, maximum number of iteration
        tol : float, tolerance to stop convergence
        clustering_n_init : int, number of times the KMeans algorithms will
            be run with different centroid seeds (n_init for KMeans)
        clustering_init : {'estimate' or 'k-means++'}, initialization method
            for KMeans. 'k-means++' is the standard for KMeans while 'estimate'
            will use the centroids from the previous step
        metric : ther metric to use for scoring, either 'silhouette' or 'CH'

        Attributes
        ---
        best_k_ : best candidate for the number of lines
        best_model_ : the best KLines model
        scores_ : a list of each candidate model scores
        """
        self.candidates = candidates
        self.n_candidates = len(candidates)
        self.alpha0 = alpha0
        self.init_alpha = init_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.clustering_n_init = clustering_n_init
        self.clustering_init = clustering_init
        self.metric = metric
        self.verbose = verbose

        self.best_k_ = None
        self.best_model_ = None
        self.scores_ = None

        self._alpha0 = self.alpha0

        # check inputs
        if not(metric in ["silhouette", "CH"]):
            raise ValueError("metric should be 'silhouette' or 'CH'")
        if not(self.clustering_init in ["k-means++", "estimate"]):
            raise ValueError("clustering_init should be either 'k-means++' or 'estimate' - got {} instead".format(clustering_init))


    def fit(self, X):
        """
        Fitting algorithm: fit a KLines on X for each candidate k

        Parameters
        ---
        X : array-like, shape (n_samples, n_features)

        Returns
        ---
        best_model : a fitted KLines model that corresponds to the highest score
        best_k : the candidate k yielding the best model
        """
        self.scores_ = []
        sub_verbose = max(0, self.verbose-1)
        models = {} 

        # alpha_initializations
        if self.init_alpha is None or self.init_alpha == "update":
            alpha_initializations = [False] * self.n_candidates
        elif self.init_alpha == "one":
            alpha_initializations = [True] + [False] * (self.n_candidates - 1)
        elif self.init_alpha == "all":
            alpha_initializations = [True] * self.n_candidates
        else:
            raise ValueError("init_alpha must be None, 'one', 'all' or 'update' - got {} instead".format(self.init_alpha))

        # update_alpha0 after the first candidate if necessary
        if self.init_alpha == "one" or self.init_alpha == "update":
            update_alpha0 = True
        else:
            update_alpha0 = False

        # maximum number of iterations for each candidate
        if type(self.max_iter) is int:
            max_iterations = [self.max_iter] * self.n_candidates
        elif len(self.max_iter) != self.n_candidates:
            raise ValueError("max_iter should be an integer or a list of the same size of the number of candidates")
        else:
            max_iterations = self.max_iter


        # fit a KLines model for each candidate k
        for idx, k in enumerate(self.candidates):
            if self.verbose > 1:
                print(("-Test {} components".format(k)))

            model = KLines(k, 
                alpha0=self._alpha0, init_alpha=alpha_initializations[idx], 
                max_iter=max_iterations[idx], tol=self.tol, 
                clustering_n_init=self.clustering_n_init, clustering_init=self.clustering_init, 
                metric=self.metric, verbose=sub_verbose
            )
            model.fit(X)
            models[k] = model
            self.scores_.append(model.score())

             # update_alpha0 after the first candidate if necessary
            if update_alpha0:
                self._alpha0 = model.alpha_
                update_alpha0 = False

        # select best model
        self.best_k_ = self.candidates[np.argmax(self.scores_)]
        self.best_model_ = models[self.best_k_]

        if self.verbose:
            print("-Results:")
            print(("Candidate scores: {}".format(self.scores_)))
            print(("Best model with {} components".format(self.best_k_)))
            print(("Best model with orientation: {:.2f}".format(utils.rad2deg(self.best_model_.alpha_))))
            
        return self.best_model_, self.best_k_


    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to

        Parameters
        ---
        X : array-like, shape (n_samples, n_features)

        Returns
        ---
        labels : array, index of the cluster each sample belongs to
        """
        if self.best_model_ is None:
            raise RuntimeError("Model needs to be fitted first")

        return self.best_model_.predict(X)
