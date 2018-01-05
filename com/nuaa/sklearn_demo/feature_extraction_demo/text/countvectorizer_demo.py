from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

corpus = ['海滨 岛屿 城市观光 购物血拼 酒店', '婚纱摄影 美食 旅游婚礼 酒店', '古迹遗址 度假酒店 湖光山色 酒店','海滨 岛屿 古迹遗址  酒店',' 岛屿 美食 旅游婚礼 城市观光 购物血拼 酒店']
vectorizer = CountVectorizer()
corpusTotoken = vectorizer.fit_transform(corpus).todense()
print(corpusTotoken)
print(vectorizer.vocabulary_)
