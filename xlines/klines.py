import utils
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabaz_score


class KLines(object):
    
    def __init__(self, n_components, metric="silhouette", verbose=2):
        """
        Initialize the KLines object

        Parameters
        ---
        n_components : int, number of lines to fit
        metric : ther metric to use for scoring, either 'silhouette' or 'CH'
        """
        self.alpha = None
        self.n_components = n_components
        self.metric = metric
        self.verbose = verbose
        
        self._labels = None
        self._centers = None
        self._Xprojected = None

        self.clustering_n_init_ = None
        self.clustering_init_ = None
        self.score_ = None

        # check inputs
        if not(metric in ["silhouette", "CH"]):
            raise ValueError("metric should be 'silhouette' or 'CH'")


    def _cluster(self, X, alpha, store, verbose):
        """
        Project data X on the normal direction and cluster with KMeans

        Parameters
        ---
        X : array-like, shape (n_samples, n_features)
        alpha : float, orientation to project
        store : boolean, if True store the projection and cluster labels

        Returns
        ---
        labels : the list of cluster index in the same order as X
        score : opposite of the clustering inertia (higher the better)
        """
        R = utils.rotation_matrix(-alpha)
        rotX = np.dot(R, X.T).T
        projX = rotX[:,1].reshape((-1,1))

        if verbose > 1:
            plt.scatter(rotX[:,0], rotX[:,1])
            plt.show()


        if self.clustering_init_ == "estimate" and self._centers is not None:
            init_centers = np.dot(R, self._centers.T).T
            init_centers = init_centers[:,1].reshape(-1, 1)
            n_init = 1
        else:
            init_centers = "k-means++"
            n_init = self.clustering_n_init_

        alg = KMeans(n_clusters=self.n_components, init=init_centers, n_init=n_init)
        alg.fit(projX)
        labels = alg.predict(projX)
        score = -alg.inertia_ # alg.score(projX)


        if store:
            self._labels = labels
            self._Xprojected = projX
        
        if self.clustering_init_ == "estimate" or store:
            projCenters = np.zeros((self.n_components, 2))
            projCenters[:,1] = alg.cluster_centers_[:,0]
            self._centers = np.dot(R.T, projCenters.T).T

        if verbose:
            print("Kmeans w/ proj {}: {}".format(utils.rad2deg(alpha), score))
        
        return labels, score


    def _init_alpha(self, X):
        """
        Initialization step to guess the best alpha0

        Parameters
        ---
        X : array-like, shape (n_samples, n_features)

        Returns
        ---
        alpha : selected orientation
        """
        alphas = [utils.deg2rad(a) for a in range(-80, 90, 40)]
        _, scores = zip(*[self._cluster(X, a, store=False, verbose=0) for a in alphas])
        alpha = alphas[np.argmax(scores)]
        
        if self.verbose:
            print("Selected alpha={} at init".format(utils.rad2deg(alpha)))

        return alpha


    def _fit_step(self, X):
        """
        One step of the fitting algorithm: project, cluster, update alpha

        Parameters
        ---
        X : array-like, shape (n_samples, n_features)

        Returns
        ---
        alpha_diff : orientation update
        """
        
        labels, score = self._cluster(X, self.alpha, store=True, verbose=self.verbose)

        orientations = []
        for c in xrange(self.n_components):
            if np.sum(labels == c) > 1:
                a = utils.pca_orientation(X[labels == c,:])
                orientations.append(a)

        alpha = np.mean(orientations)
        alpha_diff = abs(alpha-self.alpha)
        self.alpha = alpha
        return alpha_diff


    def fit(self, X, alpha0=0., init_alpha=True, max_iter=10, tol=0.001, clustering_n_init=3, clustering_init="estimate"):
        """
        Fitting algorithm: cluster data X in K lines

        Parameters
        ---
        X : array-like, shape (n_samples, n_features)
        alpha0 : float, initial value for alpha if init_alpha is False
        init_alpha : boolean, if True test a few alpha0  and initialize 
            the iterations with the best one
        max_iter : int, maximum number of iteration
        tol : float, tolerance to stop convergence
        clustering_n_init : int, number of times the KMeans algorithms will
            be run with different centroid seeds (n_init for KMeans)
        clustering_init : {'estimate' or 'k-means++'}, initialization method
            for KMeans. 'k-means++' is the standard for KMeans while 'estimate'
            will use the centroids from the previous step

        Returns
        ---
        alpha : orientation in radians
        """

        # init
        X = np.asarray(X)
        self.score_ = None
        self.clustering_n_init_ = clustering_n_init
        self.clustering_init_ = clustering_init
        if not(self.clustering_init_ in ["k-means++", "estimate"]):
            raise ValueError("clustering_init should be either 'k-means++' or 'estimate' - got {} instead".format(clustering_init))


        # initialize alpha
        if init_alpha:
            self.alpha = self._init_alpha(X)
        else:
            self.alpha = alpha0


        alpha_diff = 100. + tol # something larger than tol
        n_iter = 0
        while(alpha_diff > tol and n_iter < max_iter):
            alpha_diff = self._fit_step(X)
            n_iter += 1

        if alpha_diff > tol and self.verbose:
            print("[Warning KLines {}] Fit did not converge but maximum number of iteration reached".format(self.n_components))

        return self.alpha


    def score(self):
        """
        Score of the KMeans clustering on the projected axis
        """
        if self.score_ is None and self._labels is not None:
            if self.metric == "CH":
                self.score_ = calinski_harabaz_score(self._Xprojected, self._labels)
            elif self.metric == "silhouette":
                self.score_ = silhouette_score(self._Xprojected, self._labels, metric="euclidean")
            else:
                self.score_ = None
        return self.score_


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

        if self._centers is None:
            raise RuntimeError("Model needs to be fitted first")

        R = utils.rotation_matrix(-self.alpha)
        projX = (np.dot(R, np.asarray(X).T).T)[:,1]
        projCentroids = (np.dot(R, self._centers.T).T)[:,1]

        labels = [np.argmin([abs(x - c) for c in projCentroids]) for x in projX]

        return labels
