
from Dataset import MyDataset

npz_dir = "/Users/jinchenji/Developer/Datasets/healthcare/Article3/npz_files"
csv_path = "/Users/jinchenji/Developer/Datasets/healthcare/Article3/Data.csv"

dataset = MyDataset(npz_dir)
dataset.export_dataset(csv_path, hard_mode=False)



# import pandas as pd
# import numpy as np
# import os.path
# import random
# import shutil
# from feature_token import structural_feature_list, semantic_feature_list
#
# from tqdm import tqdm
#
# dataset_path = "/Users/jinchenji/Developer/Datasets/healthcare/Article3/Data.csv"
# save_dir = "/Users/jinchenji/Developer/Datasets/healthcare/Article3/npz_files"
#
# csv_data = pd.read_csv(dataset_path)
# data_length = len(csv_data)
#
#
# for idx in tqdm(range(data_length)):
#     label = csv_data["Symptom Mistakes"][idx]
#     if np.isnan(label):
#         label = 0.0
#     structural_feature = np.zeros(len(structural_feature_list))
#     for i, token in enumerate(structural_feature_list):
#         structural_feature[i] = csv_data[token][idx]
#
#     semantic_feature = np.zeros(len(semantic_feature_list))
#     for i, token in enumerate(semantic_feature_list):
#         semantic_feature[i] = csv_data[token][idx]
#
#     np.savez(os.path.join(save_dir, "{}.npz".format(idx)),
#              structural_feature=structural_feature, semantic_feature=semantic_feature,
#              label=label)
#
#
# train_sample_count = [76, 53]
# while train_sample_count[0] + train_sample_count[1] > 0:
#     r_int = random.randint(0, 184)
#     fn = os.path.join(save_dir, "{}.npz".format(r_int))
#     if os.path.exists(fn):
#         label = 0 if np.load(fn)["label"] == 0 else 1
#         if train_sample_count[label] > 0:
#             shutil.move(fn, os.path.join(save_dir, "train"))
#             train_sample_count[label] -= 1
#         else:
#             shutil.move(fn, os.path.join(save_dir, "test"))