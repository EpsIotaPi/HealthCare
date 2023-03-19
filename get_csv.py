import pandas as pd
import numpy as np
from Article3.Dataset.feature_token import all_features_list

dataset_path = "/Users/jinchenji/Developer/Datasets/healthcare/Article3/Data.csv"


csv_data = pd.read_csv(dataset_path)
data_length = len(csv_data)



for idx in range(data_length):
    # print(csv_data["Symptom Mistakes"][idx])
    # print(type(csv_data["Symptom Mistakes"][idx]))
    if np.isnan(csv_data["Symptom Mistakes"][idx]):
        print("godd")
