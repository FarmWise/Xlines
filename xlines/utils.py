"""
Utils for KLines and XLines
"""

import numpy as np
from sklearn.decomposition import PCA


def rad2deg(a):
    return (a*180.)/np.pi


def deg2rad(a):
    return (a*np.pi)/180.


def rotation_matrix(a, radians=True):
    alpha = a if radians else (np.pi * a / 180.)
    c = np.cos(alpha)
    s = np.sin(alpha)
    return np.array([[c, -s], [s, c]])


def pca_orientation(X):
    pca = PCA(n_components=1).fit(X)
    v = pca.components_[0]
    return np.arctan(v[1] / v[0])
