import os
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

fh = open('./youji_ids_content_exact_top_words')

ids = []
corpus = []

for line in fh.readlines():
    obj = json.loads(line)
    try:
        if len(obj['contents']) > 0:
            ids.append(obj['id'])
            corpus.append(" ".join(obj['contents']))
    except Exception as e:
        print(line)

print(len(ids))
print(len(corpus))

vectorizer = CountVectorizer()
corpusTotoken = vectorizer.fit_transform(corpus).todense()
print(corpusTotoken)

dict = vectorizer.vocabulary_
print(dict)

tf_transformer = TfidfTransformer().fit(corpusTotoken)
X_train_tf = tf_transformer.transform(corpusTotoken).todense()
all_similarity = (X_train_tf * X_train_tf.T).A

similarity = []

for index in range(all_similarity.shape[0]):
    single_simulator_top_20 = np.argsort(all_similarity[index, :])[-20:]
    index_similarity = [ids[index] for index in single_simulator_top_20]
    index_similarity.reverse()
    similarity.append(index_similarity)

file_object = open('similarity.txt', 'a+')
for index in range(len(similarity)):
    file_object.write(" ".join(similarity[index]))
    file_object.write("\n")
file_object.close()
