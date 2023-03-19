
import os
import numpy as np

from sklearn.svm import SVC
# from skrvm import RVC
# from sklearn.naive_bayes import MultinomialNB as MNB
# from sklearn.model_selection import cross_validate, cross_val_score, KFold
# from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, balanced_accuracy_score
from sklearn.feature_selection import RFE, RFECV
from Dataset.feature_token import structural_feature_list, semantic_feature_list, all_features_list

npz_dir = "/Users/jinchenji/Developer/Datasets/healthcare/Article3/npz_files"
train_dir = "/Users/jinchenji/Developer/Datasets/healthcare/Article3/npz_files/train"
test_dir = "/Users/jinchenji/Developer/Datasets/healthcare/Article3/npz_files/test"


featrue = "structural"  # structural (20), semantic (115), all (135)
select_set = "both"  # train, test, both
cross_validate = True
print_support = False


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


print("use feature:", featrue)
if featrue == "structural":
    X = structural_X
    X_test = structural_X_test
    feature_list = structural_feature_list
elif featrue == "semantic":
    X = semantic_X
    X_test = semantic_X_test
    feature_list = semantic_feature_list
elif featrue == "all":
    X = np.concatenate([structural_X, semantic_X], axis=1)
    X_test = np.concatenate([structural_X_test, semantic_X_test], axis=1)
    feature_list = all_features_list

print("use set    :", select_set)
if select_set == "test":
    X = X_test
    y = y_test
elif select_set == "all":
    X = np.concatenate([X, X_test], axis=0)
    y = np.concatenate([y, y_test], axis=0)

print(X.shape)
print("-" * 30)

estimator = SVC(kernel="linear")

if cross_validate:
    selector = RFECV(estimator).fit(X, y)
else:
    selector = RFE(estimator, n_features_to_select=14).fit(X, y)


feature_support = []
for i in range(len(selector.support_)):
    if selector.support_[i]:
        feature_support.append(feature_list[i])
print(feature_support, len(feature_support))
if print_support:
    print(selector.support_)