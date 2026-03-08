from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def apply_pca(X_train, X_test, n_components):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)


    return X_train_pca, X_test_pca, pca

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def apply_pca(X_train, X_test, n_components: int):
“””
Scale features with StandardScaler then reduce dimensionality with PCA.
The scaler and PCA are fitted on X_train only; X_test is transformed
using those fitted objects to prevent data leakage.

Parameters
----------
X_train      : training feature matrix (array-like)
X_test       : test feature matrix (array-like)
n_components : number of principal components to retain

Returns
-------
X_train_pca  : transformed training matrix
X_test_pca   : transformed test matrix
pca          : fitted PCA object (useful for explained variance inspection)
"""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)[-1]
print(
    f"[apply_pca] {n_components} components retain "
    f"{cumulative_variance * 100:.2f}% of variance."
)

return X_train_pca, X_test_pca, pca
