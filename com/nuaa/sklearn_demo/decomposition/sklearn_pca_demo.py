import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs

# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本3个特征，共4个簇
X, y = make_blobs(n_samples=10000, n_features=3, centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
                  cluster_std=[0.2, 0.5, 0.2, 0.2],
                  random_state=9)
# fig = plt.figure()
# ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
# plt.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o')

from sklearn.decomposition import PCA

import numpy as np

pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

X_new = pca.transform(X)
# plt.scatter(X_new[:, 0], X_new[:, 1],marker='o')
# plt.show()


pca = PCA(n_components=0.95)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)

pca = PCA(n_components=0.99)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)

import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1, -1, 1], [-2, -1, 1], [-3, -2, 1], [1, 1, 1], [2, 1, 0], [3, 2, 0]])
pca = PCA(n_components=3)
pca.fit(X)
variance_ration = pca.explained_variance_ratio_
sort = np.argsort(variance_ration)
print(sort)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
