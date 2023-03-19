
import os
import numpy as np

from sklearn.svm import SVC
from skrvm import RVC
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.model_selection import cross_validate, cross_val_score, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, balanced_accuracy_score
from sklearn.feature_selection import RFE, RFECV
from Dataset.feature_token import structural_feature_list, semantic_feature_list, all_features_list

npz_dir = "/Users/jinchenji/Developer/Datasets/healthcare/Article3/npz_files"
train_dir = "/Users/jinchenji/Developer/Datasets/healthcare/Article3/npz_files/train"
test_dir = "/Users/jinchenji/Developer/Datasets/healthcare/Article3/npz_files/test"


featrue = "semantic"  # structural (20), semantic (115), all (135)
model_name = "rvm"    # rvm, mnb
# feature_num = 115


structural_X = np.zeros([129, 20])
semantic_X = np.zeros([129, 115])
y = np.zeros([129,])

train_list = os.listdir(train_dir)
for idx, fn in enumerate(train_list):
    npz_sample = np.load(os.path.join(train_dir, fn))
    structural_X[idx] = npz_sample["structural_feature"]
    semantic_X[idx] = npz_sample["semantic_feature"]
    y[idx] = 0 if npz_sample["label"] == 0 else 1


structural_X_test = np.zeros([56, 20])
semantic_X_test = np.zeros([56, 115])
y_test = np.zeros([56,])

test_list = os.listdir(test_dir)
for idx, fn in enumerate(test_list):
    npz_sample = np.load(os.path.join(test_dir, fn))
    structural_X_test[idx] = npz_sample["structural_feature"]
    semantic_X_test[idx] = npz_sample["semantic_feature"]
    y_test[idx] = 0 if npz_sample["label"] == 0 else 1


estimator = SVC(kernel="linear")
# selector = RFE(estimator).fit(structural_X, y)
selector = RFECV(estimator).fit(structural_X, y)
for i in range(len(selector.support_)):
    if selector.support_[i]:
        print(structural_feature_list[i])

print("use feature:", featrue)
if featrue == "structural":

    X = structural_X
    X_test = structural_X_test
elif featrue == "semantic":
    X = semantic_X
    X_test = semantic_X_test
elif featrue == "all":
    X = np.concatenate([structural_X, semantic_X], axis=1)
    X_test = np.concatenate([structural_X_test, semantic_X_test], axis=1)

print("use method:", model_name)
if model_name == "mnb":
    clf = MNB()
elif model_name == "rvm":
    clf = RVC(n_iter=10000)
print("-"*30)

#
# scoring = ['roc_auc', 'accuracy', 'f1', 'recall', 'balanced_accuracy']
# results = cross_validate(clf, X, y, cv=5, return_estimator=True, scoring=scoring, return_train_score=True)
#
# print("Train AUC   :", results["train_roc_auc"], results["train_roc_auc"].mean(), results["train_roc_auc"].std())
# print("")
# print("AUC         :", results["test_roc_auc"], results["test_roc_auc"].mean())
# print("Accuracy    :", results["test_accuracy"], results["test_accuracy"].mean())
# print("F-Score     :", results["test_f1"], results["test_f1"].mean())
# print("Sensitivity :", results["test_recall"], results["test_recall"].mean())
# TNR = 2 * results["test_balanced_accuracy"] - results["test_recall"]
# print("Specificity :", TNR, TNR.mean())
#
# print("-"*30)
#
# length = len(results["estimator"])
# train_auc = np.zeros(length)
# auc = np.zeros(length)
# accuracy = np.zeros(length)
# f1 = np.zeros(length)
# sensitivity = np.zeros(length)
# specificity = np.zeros(length)
#
# for idx, clf in enumerate(results["estimator"]):
#     train_auc[idx] = roc_auc_score(y, clf.predict(X))
#
#     pred = clf.predict(X_test)
#     auc[idx] = roc_auc_score(y_test, pred)
#     accuracy[idx] = accuracy_score(y_test, pred)
#     f1[idx] = f1_score(y_test, pred)
#     sensitivity[idx] = recall_score(y_test, pred)
#     balance_acc = balanced_accuracy_score(y_test, pred)
#     specificity[idx] = 2 * balance_acc - sensitivity[idx]
#
#     # auc[idx] = roc_auc_score(y_test, clf.predict(X_test))
#     # accuracy[idx] = clf.score(X_test, y_test)
#
# print("Train_AUC   :", train_auc, train_auc.mean(), train_auc.std())
# print("")
# print("AUC         :", auc, auc.mean())
# print("Accuracy    :", accuracy, accuracy.mean())
# print("F-Score     :", f1, f1.mean())
# print("Sensitivity :", sensitivity, sensitivity.mean())
# print("Specificity :", specificity, specificity.mean())
