import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline

# 1. Load data
input_file = 'models/train.csv'
output_file = 'models/train_normalized.csv'
df = pd.read_csv(input_file)

# 2. Select and prepare numeric features (mirroring ModelTrainer logic)
X_numeric = df.select_dtypes(include=[np.number]).fillna(0)
cols_to_drop = ["tournament_year", "stage_name"]
X_numeric = X_numeric.drop(columns=[c for c in cols_to_drop if c in X_numeric.columns], errors="ignore")
const_cols = [c for c in X_numeric.columns if X_numeric[c].nunique() <= 1 or X_numeric[c].var() < 1e-8]
X_numeric = X_numeric.drop(columns=const_cols, errors='ignore')

# 3. Define the same pipeline used in ModelTrainer
pipeline = Pipeline([
    ('robust', RobustScaler()),
    ('std', StandardScaler())
])

# 4. Transform
X_scaled = pipeline.fit_transform(X_numeric)

# 5. Create DataFrame and save
X_scaled_df = pd.DataFrame(X_scaled, columns=X_numeric.columns)
# Optionally add 'y' back if desired
if 'y' in df.columns:
    X_scaled_df['y'] = df['y']

X_scaled_df.to_csv(output_file, index=False)
print(f"Normalized data successfully saved to {output_file}")
