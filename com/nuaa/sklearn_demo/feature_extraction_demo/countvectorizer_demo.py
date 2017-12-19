from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
corpus=['常见 的 方法 就 那么 几种','相差 不会 很大','如果 需要 达到 很高 的 分数 还 需要 算法 上 的 修改 优化']
vectorizer=CountVectorizer()
corpusTotoken=vectorizer.fit_transform(corpus).todense()
print(corpusTotoken)

tf_transformer = TfidfTransformer().fit(corpusTotoken)
X_train_tf = tf_transformer.transform(corpusTotoken)
print(X_train_tf)