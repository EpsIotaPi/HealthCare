
import os, random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale, normalize

from .feature_token import structural_feature_list, semantic_feature_list
from .feature_support import select_features, supports_dict



class MyDataset():
    def __init__(self, npz_dir):
        self.npz_dir = npz_dir
        self.train_dir = os.path.join(npz_dir, "train")
        self.test_dir = os.path.join(npz_dir, "test")

        # Set the number of negative/positive samples in the train/test dataset. According to the paper.
        self.train_neg, self.train_pos = 76, 53
        self.test_neg, self.test_pos = 34, 22
        self.train_sample_num = self.train_pos + self.train_neg  # len(os.listdir(self.train_dir))
        self.test_sample_num = self.test_pos + self.test_neg    # len(os.listdir(self.test_dir))

    def train_test_set(self, use_featrue="structural", use_support=False):
        structural_X = np.zeros([self.train_sample_num, 20])
        semantic_X = np.zeros([self.train_sample_num, 115])
        y = np.zeros([self.train_sample_num, ])
        for idx, fn in enumerate(os.listdir(self.train_dir)):
            npz_sample = np.load(os.path.join(self.train_dir, fn))
            structural_X[idx] = npz_sample["structural_feature"]
            semantic_X[idx] = npz_sample["semantic_feature"]
            y[idx] = 0 if npz_sample["label"] == 0 else 1


        structural_X_test = np.zeros([self.test_sample_num, 20])
        semantic_X_test = np.zeros([self.test_sample_num, 115])
        y_test = np.zeros([self.test_sample_num, ])
        for idx, fn in enumerate(os.listdir(self.test_dir)):
            npz_sample = np.load(os.path.join(self.test_dir, fn))
            structural_X_test[idx] = npz_sample["structural_feature"]
            semantic_X_test[idx] = npz_sample["semantic_feature"]
            y_test[idx] = 0 if npz_sample["label"] == 0 else 1

        if use_featrue == "structural":
            if use_support:
                structural_X = select_features(structural_X, support=supports_dict["TOF"])
                structural_X_test = select_features(structural_X_test, support=supports_dict["TOF"])
            return structural_X, y, structural_X_test, y_test
        elif use_featrue == "semantic":
            if use_support:
                semantic_X = select_features(semantic_X, support=supports_dict["TOF"])
                semantic_X_test = select_features(semantic_X_test, support=supports_dict["TOF"])
            return semantic_X, y, semantic_X_test, y_test
        else:
            X = np.concatenate([semantic_X, structural_X], axis=1)
            X_test = np.concatenate([semantic_X_test, structural_X_test], axis=1)
            if use_support:
                X = select_features(X, support=supports_dict["JOF"])
                X_test = select_features(X_test, support=supports_dict["JOF"])
            return X, y, X_test, y_test

    def load_set(self, use_set="train", use_featrue="structural", norm_method="none"):
        X, y, X_test, y_test = self.train_test_set(use_featrue=use_featrue)

        if norm_method == "minmax":
            X = minmax_scale(X)
            X_test = minmax_scale(X_test)
        elif norm_method == "l2":
            X = normalize(X, norm='l2')
            X_test = normalize(X_test, norm='l2')

        if use_set == "train":
            return X, y
        elif use_set == "test":
            return X_test, y_test
        else:
            X_all = np.concatenate([X, X_test], axis=0)
            y_all = np.concatenate([y, y_test], axis=0)
            return X_all, y_all

    def export_dataset(self, csv_file, hard_mode=False):  # csv_file = =".../Article3/Data.csv"

        if not hard_mode:
            if os.path.exists(self.train_dir) or os.path.exists(self.test_dir):
                print("Dataset Dir Already Exist")
                print("Please Check {} and {}".format(self.train_dir, self.test_dir))
                return

        os.makedirs(self.train_dir, exist_ok=hard_mode)
        os.makedirs(self.test_dir, exist_ok=hard_mode)

        train_sample_count = [self.train_neg, self.train_pos]
        test_sample_count = [self.test_neg, self.test_pos]

        csv_data = pd.read_csv(csv_file)
        data_length = len(csv_data)

        for idx in tqdm(range(data_length)):
            label = csv_data["Symptom Mistakes"][idx]
            sample_label = 1
            if np.isnan(label):
                label = 0.0
                sample_label = 0
            structural_feature = np.zeros(len(structural_feature_list))
            for i, token in enumerate(structural_feature_list):
                structural_feature[i] = csv_data[token][idx]

            semantic_feature = np.zeros(len(semantic_feature_list))
            for i, token in enumerate(semantic_feature_list):
                semantic_feature[i] = csv_data[token][idx]

            r_int = random.randint(0, 1)
            if r_int == 0:
                if train_sample_count[sample_label] > 0:
                    save_dir = self.train_dir
                    train_sample_count[sample_label] -= 1
                else:
                    save_dir = self.test_dir
                    test_sample_count[sample_label] -= 1
            elif r_int == 1:
                if test_sample_count[sample_label] > 0:
                    save_dir = self.test_dir
                    test_sample_count[sample_label] -= 1
                else:
                    save_dir = self.train_dir
                    train_sample_count[sample_label] -= 1
            if train_sample_count[sample_label] < 0 or test_sample_count[sample_label] < 0:
                print(train_sample_count, test_sample_count)
                print("The count of dataset sample have error")
                return

            np.savez(os.path.join(save_dir, "{}.npz".format(idx)),
                     structural_feature=structural_feature,
                     semantic_feature=semantic_feature,
                     label=label)
