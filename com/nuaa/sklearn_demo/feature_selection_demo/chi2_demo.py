from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np

# 与sklearn_svm_isas_demo.py对比，这边先进行特征提取，然后在用SVM进行分类算法

iris = load_iris()
iris_data, iris_target = iris.data, iris.target

print(iris_data.shape)
new_data = SelectKBest(chi2, k=3).fit_transform(iris_data, iris_target)
print(new_data.shape)
#new_data = StandardScaler().fit_transform(new_data)

# 拼接成150*5的矩阵，进行乱序
iris_concatenate_data = np.concatenate((new_data, np.array(iris_target).reshape(new_data.shape[0], -1)), axis=1)
np.random.shuffle(iris_concatenate_data)

# 重新获取数据
iris_data = iris_concatenate_data[:, :-1]
iris_target = iris_concatenate_data[:, -1]

traing_data = iris_data[0:120]
traing_target = iris_target[0:120]

test_data = iris_data[120:150]
test_target = iris_target[120:150]

rbf_svc = SVC()
rbf_svc.fit(traing_data, traing_target)

predict = rbf_svc.predict(test_data)

# 获取标签和对应的分类名称
labels = [0, 1, 2]
target_names = iris.target_names
print(metrics.classification_report(test_target, predict, labels, target_names))
