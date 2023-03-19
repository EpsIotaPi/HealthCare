import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, GridSearchCV, KFold, StratifiedKFold, cross_val_score
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


# specify range of hyperparameters
hyper_params = [{
                "n_features_to_select": [0.25, 0.5, 0.75, 1.0],
                "estimator__C": [1000, 100],
                "estimator__epsilon": [0.1, 0.001, 0]}]

# folds = KFold(n_splits=5, shuffle = True, random_state=100)



svc = SVC(kernel="linear")
# for n_f in range(1, 118):
#     X_select = np.zeros([235, n_f])
#     idx_c = 0
#     rfecv = RFE(estimator=svc, n_features_to_select=n_f)
#     rfecv.fit(X, y)
#     for idx, s in enumerate(rfecv.support_):
#         if s:
#             X_select[:, idx_c] = X[:, idx]
#             idx_c += 1
#
#     results = cross_val_score(svc, X_select, y, cv=5)
#     print(n_f, results.mean())


rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X, y)
print("Optimal number of features : %d" % rfecv.n_features_)
print("Ranking of features : %s" % rfecv.ranking_)
print("Ranking of features : %s" % rfecv.support_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1), rfecv.cv_results_["mean_test_score"])
plt.show()
