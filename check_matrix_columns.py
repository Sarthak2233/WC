import pandas as pd
from src.features.csv_oracle import CSVFeatureOracle

oracle_engine = CSVFeatureOracle("data/processed")
print(oracle_engine.unified_features.columns)
print(oracle_engine.unified_features[oracle_engine.unified_features['year'] == 2026]['canonical_team'].unique())
