from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np

# 与sklearn_svm_isas_demo.py对比，这边先进行特征提取，然后在用SVM进行分类算法

iris = load_iris()
iris_data, iris_target = iris.data, iris.target

print(iris_data.shape)
new_data = SelectKBest(chi2, k=3).fit_transform(iris_data, iris_target)
print(new_data.shape)
new_data = StandardScaler().fit_transform(new_data)

traing_data,test_data, traing_target, test_target = train_test_split(new_data,
                                                                     iris_target,
                                                   test_size = 0.2)

rbf_svc = SVC()
rbf_svc.fit(traing_data, traing_target)

predict = rbf_svc.predict(test_data)

# 获取标签和对应的分类名称
labels = [0, 1, 2]
target_names = iris.target_names
print(metrics.classification_report(test_target, predict, labels, target_names))
