from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder

import numpy as np

data = load_iris().data[0:2, ]
target = load_iris().target

print(data)
print(StandardScaler().fit_transform(data))
print(MinMaxScaler().fit_transform(data))
print(Normalizer().fit_transform(data))

print(Binarizer(threshold=3).fit_transform(data))
print(OneHotEncoder().fit_transform(target.reshape((-1, 1))))

print(target)

from sklearn.feature_selection import VarianceThreshold

print(VarianceThreshold(threshold=0.5).fit_transform(load_iris().data))
print(np.std(load_iris().data, axis=0))

from sklearn.decomposition import PCA

print(PCA(n_components=2).fit_transform(load_iris().data))
