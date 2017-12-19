print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import metrics

data, target = make_multilabel_classification(n_samples=1000, n_classes=3, n_labels=3,
                                              allow_unlabeled=True,
                                              random_state=1)

traing_data = data[0:800]
traing_target = target[0:800]

test_data = data[800:1000]
test_target = target[800:1000]

classif = OneVsRestClassifier(SVC(kernel='linear'))
classif.fit(traing_data, traing_target)

predicted = classif.predict(test_data)

print(metrics.classification_report(test_target, predicted,
                                    target_names="test"))
