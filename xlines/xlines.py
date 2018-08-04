
from klines import KLines
import utils
import numpy as np


class XLines(object):
    
    def __init__(self, candidates, metric="silhouette", verbose=2):
        """
        Initialize a XLines object

        Parameters
        ---
        candidates : array-like, list of candidate k for which we will try to fit a KLines
        metric : ther metric to use for scoring, either 'silhouette' or 'CH'
        """
        self.candidates = candidates
        self.n_candidates = len(candidates)
        self.metric = metric
        self.models = None
        self.verbose = verbose

        self._scores = None

        self.best_k_ = None
        self.best_model_ = None

        # check inputs
        if not(metric in ["silhouette", "CH"]):
            raise ValueError("metric should be 'silhouette' or 'CH'")


    def fit(self, X, alpha0=0., init_alpha="one", max_iter=10, tol=0.001, clustering_n_init=3, clustering_init="estimate"):
        """
        Fitting algorithm: fit a KLines on X for each candidate k

        Parameters
        ---
        X : array-like, shape (n_samples, n_features)
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

        Returns
        ---
        best_model : a fitted KLines model that corresponds to the highest score
        best_k : the candidate k yielding the best model
        """

        # reset scores
        self._scores = []

        # initialize models
        sub_verbose = max(0, self.verbose-1)
        self.models = dict((k, KLines(k, metric=self.metric, verbose=sub_verbose)) for k in self.candidates)

        # alpha_initializations
        if init_alpha is None or init_alpha == "update":
            alpha_initializations = [False] * self.n_candidates
        elif init_alpha == "one":
            alpha_initializations = [True] + [False] * (self.n_candidates - 1)
        elif init_alpha == "all":
            alpha_initializations = [True] * self.n_candidates
        else:
            raise ValueError("init_alpha must be None, 'one', 'all' or 'update' - got {} instead".format(init_alpha))

        # update_alpha0 after the first candidate if necessary
        if init_alpha == "one" or init_alpha == "update":
            update_alpha0 = True
        else:
            update_alpha0 = False

        # maximum number of iterations for each candidate
        if type(max_iter) is int:
            max_iterations = [max_iter] * self.n_candidates
        elif len(max_iter) != self.n_candidates:
            raise ValueError("max_iter should be an integer or a list of the same size of the number of candidates")
        else:
            max_iterations = max_iter

        # fit a KLines model for each candidate k
        for idx, k in enumerate(self.candidates):

            if self.verbose > 1:
                print("-Test {} components".format(k))

            model = self.models[k]
            model.fit(
                X, alpha0=alpha0, init_alpha=alpha_initializations[idx], max_iter=max_iterations[idx], 
                tol=tol, clustering_n_init=clustering_n_init, clustering_init=clustering_init
            )
            self._scores.append(model.score())

             # update_alpha0 after the first candidate if necessary
            if update_alpha0:
                alpha0 = model.alpha
                update_alpha0 = False

        # select best model
        self.best_k_ = self.candidates[np.argmax(self._scores)]
        self.best_model_ = self.models[self.best_k_]

        if self.verbose:
            print("-Results:")
            print("Candidate scores: {}".format(self._scores))
            print("Best model with {} components".format(best_k))
            print("Best model with orientation: {:.2f}".format(utils.rad2deg(best_model.alpha)))
            
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
