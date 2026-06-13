import pandas as pd

df = pd.read_csv('models/train.csv')
print(df.describe())
