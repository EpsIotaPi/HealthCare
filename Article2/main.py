import os
import shutil

import numpy as np
from skrvm import RVC
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, cross_val_score, KFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE, RFECV

npz_dir = "/Users/jinchenji/Developer/Datasets/healthcare/Article2/npz_files"
train_dir = "/Users/jinchenji/Developer/Datasets/healthcare/Article2/npz_files/train"
test_dir = "/Users/jinchenji/Developer/Datasets/healthcare/Article2/npz_files/test"


dataset_length = 337

X = np.zeros([235, 118])
y = np.zeros([235,])

train_list = os.listdir(train_dir)
for idx, fn in enumerate(train_list):
    npz_sample = np.load(os.path.join(train_dir, fn))
    X[idx] = npz_sample["embedding"]
    y[idx] = npz_sample["label"]


X_test = np.zeros([102, 118])
y_test = np.zeros([102,])
test_list = os.listdir(test_dir)
for idx, fn in enumerate(test_list):
    npz_sample = np.load(os.path.join(test_dir, fn))
    X_test[idx] = npz_sample["embedding"]
    y_test[idx] = npz_sample["label"]

clf = RVC()
# clf = SVC()

# results = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
# print(results.mean())

results = cross_validate(clf, X, y, cv=5, return_estimator=True)

for clf in results["estimator"]:
    scores = clf.score(X_test, y_test)
    print(scores)
    auc = roc_auc_score(y_test, clf.predict(X_test))
    print(auc)