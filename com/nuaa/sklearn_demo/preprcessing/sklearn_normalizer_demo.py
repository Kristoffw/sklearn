from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder

import numpy as np

data = load_iris().data[0:2, ]
target = load_iris().target

from sklearn import preprocessing
import numpy as np
X = np.array([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])
X_scaled = preprocessing.scale(X)


print(data)
#
standardScaler=StandardScaler().fit(data)
print(standardScaler.transform(data))
print(MinMaxScaler().fit_transform(data))
print(Normalizer().fit_transform(data))

print(Binarizer(threshold=3).fit_transform(data))
print(OneHotEncoder().fit_transform(target.reshape((-1, 1))).todense())

print(target)
