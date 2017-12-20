from sklearn.svm import SVC, SVR

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split

iris_orign_datasets = datasets.load_iris()
# 现将数据标准化
iris_orign_data = StandardScaler().fit_transform(iris_orign_datasets.data)
iris_orign_target = iris_orign_datasets.target


traing_data,test_data, traing_target, test_target = train_test_split(iris_orign_data,
                                                   iris_orign_target,
                                                   test_size = 0.2)

rbf_svc = SVC()
rbf_svc.fit(traing_data, traing_target)

predict = rbf_svc.predict(test_data)

# 获取标签和对应的分类名称
labels = [0, 1, 2]
target_names = iris_orign_datasets.target_names
print(metrics.classification_report(test_target, predict, labels, target_names))
