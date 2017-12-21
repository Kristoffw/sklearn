print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split

data, target = make_multilabel_classification(n_samples=1000, n_classes=3, n_labels=3,
                                              allow_unlabeled=True,
                                              random_state=1)

traing_data,test_data, traing_target, test_target = train_test_split(data,
                                                                     target,
                                                   test_size = 0.2)

classif = OneVsRestClassifier(SVC(kernel='linear'))
classif.fit(traing_data, traing_target)

predicted = classif.predict(test_data)

print(metrics.classification_report(test_target, predicted,
                                    target_names="tes"))
