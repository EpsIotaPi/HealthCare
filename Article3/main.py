
import os
import numpy as np

from sklearn.svm import SVC
# from skrvm import RVC
from sklearn_rvm import EMRVC as RVC
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.model_selection import cross_validate, cross_val_score, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, balanced_accuracy_score
from sklearn.preprocessing import minmax_scale, normalize
from Dataset.feature_support import select_feature, feature_support

npz_dir = "/Users/jinchenji/Developer/Datasets/healthcare/Article3/npz_files"
train_dir = "/Users/jinchenji/Developer/Datasets/healthcare/Article3/npz_files/train"
test_dir = "/Users/jinchenji/Developer/Datasets/healthcare/Article3/npz_files/test"


featrue = "structural"  # structural (20), semantic (115), all (135)
use_support = True      # TOF (5),         SOF (14),       JOF (57)

model_name = "rvm"    # rvm, mnb
norm_method = "l2"  # minmax, l2, nope
cross_val = True


structural_X = np.zeros([129, 20])
semantic_X = np.zeros([129, 115])
y = np.zeros([129,])

train_list = os.listdir(train_dir)
for idx, fn in enumerate(train_list):
    npz_sample = np.load(os.path.join(train_dir, fn))
    structural_X[idx] = npz_sample["structural_feature"]
    semantic_X[idx] = npz_sample["semantic_feature"]
    # y[idx] = npz_sample["label"]
    y[idx] = 0 if npz_sample["label"] == 0 else 1


structural_X_test = np.zeros([56, 20])
semantic_X_test = np.zeros([56, 115])
y_test = np.zeros([56,])

test_list = os.listdir(test_dir)
for idx, fn in enumerate(test_list):
    npz_sample = np.load(os.path.join(test_dir, fn))
    structural_X_test[idx] = npz_sample["structural_feature"]
    semantic_X_test[idx] = npz_sample["semantic_feature"]
    # y_test[idx] = npz_sample["label"]
    y_test[idx] = 0 if npz_sample["label"] == 0 else 1


print("use feature:", featrue)
print("use support:", use_support)
X, X_test = None, None
if featrue == "structural":
    X = structural_X.copy()
    X_test = structural_X_test.copy()
    if use_support:
        X = select_feature(X, support=feature_support["TOF"])
        X_test = select_feature(X_test, support=feature_support["TOF"])
elif featrue == "semantic":
    X = semantic_X.copy()
    X_test = semantic_X_test.copy()
    if use_support:
        X = select_feature(X, support=feature_support["SOF"])
        X_test = select_feature(X_test, support=feature_support["SOF"])
elif featrue == "all":
    # X = np.concatenate([structural_X, semantic_X], axis=1)
    # X_test = np.concatenate([structural_X_test, semantic_X_test], axis=1)
    X = np.concatenate([semantic_X, structural_X], axis=1)
    X_test = np.concatenate([semantic_X_test, structural_X_test], axis=1)
    if use_support:
        X = select_feature(X, support=feature_support["JOF"])
        X_test = select_feature(X_test, support=feature_support["JOF"])

print(X.shape)

print("use method:", model_name)
clf = None
if model_name == "mnb":
    clf = MNB(alpha=1.0)
elif model_name == "rvm":
    clf = RVC(kernel="rbf", gamma="scale")
    print("use normalize:", norm_method)
    if norm_method == "minmax":
        X = minmax_scale(X)
        X_test = minmax_scale(X_test)
    elif norm_method == "l2":
        X = normalize(X, norm='l2')
        X_test = normalize(X_test, norm='l2')


print("use cv:", cross_val)
print("-"*30)
if cross_val:
    scoring = ['roc_auc', 'accuracy', 'f1', 'recall', 'balanced_accuracy']
    results = cross_validate(clf, X, y, cv=5, return_estimator=True, scoring=scoring, return_train_score=True, verbose=True)

    print("AUC         :", results["test_roc_auc"], results["test_roc_auc"].mean(), results["test_roc_auc"].std())
    print("Accuracy    :", results["test_accuracy"], results["test_accuracy"].mean())
    print("F-Score     :", results["test_f1"], results["test_f1"].mean())
    print("Sensitivity :", results["test_recall"], results["test_recall"].mean())
    TNR = 2 * results["test_balanced_accuracy"] - results["test_recall"]
    print("Specificity :", TNR, TNR.mean())
else:
    clf.fit(X, y)
    pred = clf.predict(X_test)

    auc = roc_auc_score(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    sensitivity = recall_score(y_test, pred)
    balance_acc = balanced_accuracy_score(y_test, pred)
    specificity = 2 * balance_acc - sensitivity

    print("AUC         :", auc)
    print("Accuracy    :", accuracy)
    print("F-Score     :", f1)
    print("Sensitivity :", sensitivity)
    print("Specificity :", specificity)





