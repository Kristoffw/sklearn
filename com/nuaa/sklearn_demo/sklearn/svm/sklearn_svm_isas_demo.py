from sklearn.svm import SVC, SVR

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np

iris_orign_datasets = datasets.load_iris()
# 现将数据标准化
iris_orign_data = StandardScaler().fit_transform(iris_orign_datasets.data)
iris_orign_target = iris_orign_datasets.target

# 拼接成150*5的矩阵，进行乱序
iris_concatenate_data = np.concatenate((iris_orign_data, np.array(iris_orign_target).reshape(iris_orign_data.shape[0], -1)), axis=1)
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
target_names = iris_orign_datasets.target_names
print(metrics.classification_report(test_target, predict, labels, target_names))