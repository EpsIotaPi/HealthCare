import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, RFECV

from Dataset import MyDataset
from Dataset.feature_token import feature_dict

npz_dir = "/Users/jinchenji/Developer/Datasets/healthcare/Article2/npz_files"

featrue = "usas"  # usas (115)
select_set = "all"      # train, test, all
norm_method = "none"    # none, minmax, l2
cross_val = False
print_support = True

cv = 5                    #  only effect when cross_val=True
n_features_to_select = 5  #  only effect when cross_val=False


print("use set      :", select_set)
print("use featrue  :", featrue)
print("use normalize:", norm_method)
dataset = MyDataset(npz_dir)
X, y = dataset.load_set(use_set=select_set, use_featrue=featrue, norm_method=norm_method)
print(X.shape)

print("use cross val:", cross_val)
print("-" * 30)

estimator = SVC(kernel="linear")
if cross_val:
    selector = RFECV(estimator, cv=cv).fit(X, y)
else:
    selector = RFE(estimator, n_features_to_select=n_features_to_select).fit(X, y)


select_features = []
feature_list = feature_dict[featrue]
for i in range(len(selector.support_)):
    if selector.support_[i]:
        select_features.append(feature_list[i])

print("Length =", len(select_features))
print("\nSelect Features:")
print(select_features)
if print_support:
    print("\nFeature Support List:")
    support_list = "["
    for s in selector.support_:
        support_list += "{}, ".format(s)
    support_list = support_list[:-2] + "]"
    print(support_list)


# print("Optimal number of features : %d" % selector.n_features_)
# print("Ranking of features : %s" % selector.ranking_)
# print("Ranking of features : %s" % selector.support_)
#
# # Plot number of features VS. cross-validation scores
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (nb of correct classifications)")
# plt.plot(range(1, len(selector.cv_results_["mean_test_score"]) + 1), selector.cv_results_["mean_test_score"])
# plt.show()
