import pandas as pd

path = "/Users/jinchenji/Developer/JetBrains/Pycharm/healthcare/datasets/Article2/main.csv"

csv_data = pd.read_csv(path)

# View the first 5 rows

idx = 5


print(csv_data["Label (1=Mistake present, 0= No mistake)"][idx])
print(csv_data["Text"][idx])