from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

corpus = ['海滨 岛屿 城市观光 购物血拼 酒店', '婚纱摄影 美食 旅游婚礼 酒店', '古迹遗址 度假酒店 城市观光', '海滨 岛屿 古迹遗址  酒店', ' 岛屿 美食 旅游婚礼 城市观光 购物血拼 酒店']

vectorizer = CountVectorizer()
corpusTotoken = vectorizer.fit_transform(corpus).todense()
print(corpusTotoken)

dict = vectorizer.vocabulary_
print(dict)

tf_transformer = TfidfTransformer().fit(corpusTotoken)
X_train_tf = tf_transformer.transform(corpusTotoken).todense()
print((X_train_tf * X_train_tf.T).A)

from sklearn.decomposition import PCA
import numpy as np
pca = PCA(n_components=10)
print(corpusTotoken)
pca.fit(np.array(corpusTotoken))
print(pca.explained_variance_ratio_)