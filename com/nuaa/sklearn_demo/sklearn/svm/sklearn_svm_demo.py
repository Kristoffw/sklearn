import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC, SVR
from sklearn import metrics
from sklearn.decomposition import PCA

data, target = make_blobs(n_samples=1000, n_features=20, centers=15)
pca = PCA(n_components=0.99)
pca.fit(data)
new_data = pca.transform(data)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

traing_data = new_data[0:800]
traing_target = target[0:800]

test_data = new_data[800:1000]
test_target = target[800:1000]
rbf_svc = SVC()
model = rbf_svc.fit(traing_data, traing_target)

predicted = model.predict(test_data)

print(metrics.classification_report(test_target, predicted,
                                    target_names="test"))
