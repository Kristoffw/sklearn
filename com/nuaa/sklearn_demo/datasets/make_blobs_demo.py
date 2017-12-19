from sklearn.datasets import make_blobs
from matplotlib import pyplot

data,target=make_blobs(n_samples=100,n_features=2,centers=3)

# 在2D图中绘制样本，每个样本颜色不同
pyplot.scatter(data[:,0],data[:,1],c=target);
pyplot.show()