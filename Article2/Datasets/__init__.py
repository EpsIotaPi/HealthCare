import os.path
import numpy as np
from sklearn.model_selection import train_test_split

# dataset_path = "/Users/jinchenji/Developer/Datasets/healthcare/Article2/main.csv"
#
# csv_data = pd.read_csv(dataset_path)
# data_length = len(csv_data)
#
# nlp_embedding = get_usas_feature()
#
#
# embedding = {}
#
# embed = np.load("/Users/jinchenji/Developer/Datasets/healthcare/Article2/npz_files/128.npz")["label"]
# print(embed)


# npz_dir = "/Users/jinchenji/Developer/Datasets/healthcare/Article2/npz_files"
#
# dataset_length = 337
# labels = np.zeros(dataset_length,)
# embeddings = np.zeros([dataset_length, 118])
#
# for i in range(dataset_length):
#     npz_sample = np.load(os.path.join(npz_dir, "{}.npz".format(i)))
#     labels[i] = npz_sample["label"]
#     embeddings[i] = npz_sample["embedding"]

#
# X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.4, random_state=0)
# print(embeddings.shape)
# estimator = SVR(kernel="linear")
# selector = RFE(estimator, n_features_to_select=10, step=1)
# selector = selector.fit(embeddings, labels)
#
# scores = cross_val_score(estimator, embeddings, labels, cv=5)
# print(scores)
