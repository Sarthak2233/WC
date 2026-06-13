import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('models/train.csv')

# Select only numeric features (excluding 'y')
numeric_features = df.select_dtypes(include=[np.number]).drop(columns=['y'], errors='ignore')

# Calculate skewness
skewness = numeric_features.skew().sort_values(ascending=False)

print("Feature Skewness:")
print(skewness)
