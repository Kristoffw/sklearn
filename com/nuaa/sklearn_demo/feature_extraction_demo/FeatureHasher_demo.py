from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MultiLabelBinarizer
h = FeatureHasher(n_features=10)
D = [{'dog': 1, 'cat': 2, 'elephant': 4}, {'dog': 2, 'run': 5}]
f = h.transform(D)
print(f.toarray())
mlb = MultiLabelBinarizer()
fit_result = mlb.fit_transform([(1, 2), (3,)])
print(fit_result)
print(mlb.classes_)