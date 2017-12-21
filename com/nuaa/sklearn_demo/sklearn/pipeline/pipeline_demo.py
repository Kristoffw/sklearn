from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
from sklearn.model_selection  import cross_val_predict

Q = {'What does the "yield" keyword do in Python?': ['python'],
     'What is a metaclass in Python?': ['oop'],
     'How do I check whether a file exists using Python?': ['python'],
     'How to make a chain of function decorators?': ['python', 'decorator'],
     'Using i and j as variables in Matlab': ['matlab', 'naming-conventions'],
     'MATLAB: get variable type': ['matlab'],
     'Why is MATLAB so fast in matrix multiplication?': ['performance'],
     'Is MATLAB OOP slow or am I doing something wrong?': ['matlab-oop'],
    }
dataframe = pd.DataFrame({'body': Q.keys(), 'tag': Q.values()})

mlb = MultiLabelBinarizer()
X = dataframe['body'].values[0]
lable_list = dataframe['tag'].values
y = mlb.fit_transform(lable_list[0])

classifier = Pipeline([
    ('vectorizer', CountVectorizer(lowercase=True,
                                   stop_words='english',
                                   max_df=0.8,
                                   min_df=1)),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

predicted = cross_val_predict(classifier, X, y)