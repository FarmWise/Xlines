
from klines import KLines
import utils
import numpy as np


class XLines(object):
    
    def __init__(self, candidates, verbose=2):
        """
        Initialize a XLines object
        candidates : array-like, list of candidate k for which we will try to 
            fit a KLines
        """
        self.candidates = candidates
        self.models = None
        self.verbose = verbose
        self._scores = None
        
    def fit(self, X, alpha0=0., init_alpha=True, max_iter=10, tol=0.001):
        """
        Fitting algorithm: fit a KLines on X for each candidate k
        X : array-like, shape (n_samples, n_features)
        alpha0 : float, initial value for alpha if init_alpha is False
        init_alpha : boolean, if True test a few alpha0  and initialize 
            the iterations with the best one
        max_iter : int, maximum number of iteration
        tol : float, tolerance to stop convergence
        ---
        Returns : (best_model, best_k)
        """

        self.models = dict((k, KLines(k, verbose=self.verbose-1)) for k in self.candidates)
        self._scores = []

        for k, model in self.models.iteritems():
            if self.verbose > 1:
                print("-Test {} components".format(k))
            model.fit(X, alpha0, init_alpha, max_iter, tol)
            self._scores.append(model.score())
        
        best_k = self.candidates[np.argmax(self._scores)]
        best_model = self.models[best_k]
        
        if self.verbose:
            print("-Results:")
            print("Candidate scores: {}".format(self._scores))
            print("Best model with {} components".format(best_k))
            print("Best model with orientation: {:.2f}".format(utils.rad2deg(best_model.alpha)))
            
        return best_model, best_k