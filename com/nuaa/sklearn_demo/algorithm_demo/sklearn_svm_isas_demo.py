from sklearn.svm import SVC, SVR

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

iris_data = datasets.load_iris()

new_data = StandardScaler().fit_transform(iris_data.data)

print(len(iris_data.data))
traing_data = new_data[0:120]
traing_target = iris_data.target[0:120]

test_data = new_data[120:150]
test_target = iris_data.target[120:150]

rbf_svc = SVC()
rbf_svc.fit(traing_data, traing_target)

predict = rbf_svc.predict(test_data)
print(metrics.classification_report(test_target, predict,
                                    target_names="test"))
