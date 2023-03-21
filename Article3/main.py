
from sklearn_rvm import EMRVC as RVC
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, balanced_accuracy_score
from sklearn.preprocessing import minmax_scale, normalize

from Dataset import MyDataset

npz_dir = "/Users/jinchenji/Developer/Datasets/healthcare/Article3/npz_files"

featrue = "structural"  # structural (20),        semantic (115),        all (135)
use_support = True      # TOF for structural (5), SOF for semantic (14), JOF for all (57)

model_name = "rvm"      # rvm, mnb
norm_method = "none"    # none, minmax, l2
cross_val = False       # use cross validation or not

dataset = MyDataset(npz_dir)
X, y, X_test, y_test = dataset.train_test_set(use_featrue=featrue, use_support=use_support)

print("use feature:", featrue)
print("use support:", use_support)
print(X.shape)

print("use method:", model_name)
clf = None
if model_name == "mnb":
    clf = MNB()
elif model_name == "rvm":
    clf = RVC(kernel="rbf", gamma="scale")
    # implement normalization
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
    results = cross_validate(clf, X, y, cv=5, return_estimator=True, scoring=scoring, verbose=True)

    print("AUC         :", results["test_roc_auc"], results["test_roc_auc"].mean(), results["test_roc_auc"].std())
    print("Accuracy    :", results["test_accuracy"], results["test_accuracy"].mean())
    print("F-Score     :", results["test_f1"], results["test_f1"].mean())
    print("Sensitivity :", results["test_recall"], results["test_recall"].mean())
    specificity = 2 * results["test_balanced_accuracy"] - results["test_recall"]   # use balanced accuracy to calculate specificity
    print("Specificity :", specificity, specificity.mean())
else:
    # train clf on train-set, and test it on test-set.
    clf.fit(X, y)
    pred = clf.predict(X_test)

    auc = roc_auc_score(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    sensitivity = recall_score(y_test, pred)
    balance_acc = balanced_accuracy_score(y_test, pred)     # use balanced accuracy to calculate specificity
    specificity = 2 * balance_acc - sensitivity

    print("AUC         :", auc)
    print("Accuracy    :", accuracy)
    print("F-Score     :", f1)
    print("Sensitivity :", sensitivity)
    print("Specificity :", specificity)





