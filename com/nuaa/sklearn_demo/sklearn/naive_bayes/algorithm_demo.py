from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train',
    categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test',
    categories=categories, shuffle=True, random_state=42)


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

tf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)


clf = MultinomialNB().fit(X_train_tf, twenty_train.target)

predicted = clf.predict(tf_transformer.transform(count_vect.transform(twenty_test.data)))

print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))


