
from Dataset import MyDataset

npz_dir = "/Users/jinchenji/Developer/Datasets/healthcare/Article2/npz_files"
csv_path = "/Users/jinchenji/Developer/Datasets/healthcare/Article2/main.csv"

dataset = MyDataset(npz_dir)
dataset.export_dataset(csv_path, hard_mode=False, weight_level="word", normalize=True)



# import os.path
# import random
# import shutil
#
# import numpy as np
# import pandas as pd
# from usua_features import get_usas_feature, feature_filter
# from tqdm import tqdm
#
# dataset_path = "/Users/jinchenji/Developer/Datasets/healthcare/Article2/main.csv"
# save_dir = "/Users/jinchenji/Developer/Datasets/healthcare/Article2/npz_files"
#
# csv_data = pd.read_csv(dataset_path)
# data_length = len(csv_data)
#
#
# nlp_embedding = get_usas_feature()
#
#
# # for i, word in enumerate(doc):
# #
# #     tokens = []
# #     print(word, word._.pymusas_tags)
#     # for tag in word._.pymusas_tags:
#     #     print(word._.pymusas_tags)
#
# for idx in tqdm(range(data_length)):
#     label = csv_data["Label (1=Mistake present, 0= No mistake)"][idx]
#     text = csv_data["Text"][idx]
#     feature = np.zeros(118)
#     doc = nlp_embedding(text)
#     for i, word in enumerate(doc):
#         tokens = []
#         for tag in word._.pymusas_tags:
#             tokens.extend(feature_filter(tag))
#         for t in tokens:
#             feature[t] += 1/len(tokens)
#
#     np.savez(os.path.join(save_dir, "{}.npz".format(idx)), embedding=feature / len(text), label=label, length = len(text))
#
#
# train_sample_count = [118, 117]
# while train_sample_count[0] > 0 or train_sample_count[1] > 0:
#     r_int = random.randint(0, 336)
#     fn = os.path.join(save_dir, "{}.npz".format(r_int))
#     if os.path.exists(fn):
#         label = np.load(fn)["label"]
#         if train_sample_count[label] > 0:
#             shutil.move(fn, os.path.join(save_dir, "train"))
#             train_sample_count[label] -= 1
#         else:
#             shutil.move(fn, os.path.join(save_dir, "test"))
#
#
