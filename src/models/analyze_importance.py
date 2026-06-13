import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_feature_importance():
    # 1. Load the training data (using a recent fold as proxy for importance analysis)
    # Finding the latest fold
    splits_dir = "models/data_splits"
    folds = sorted([f for f in os.listdir(splits_dir) if f.startswith("fold_")])
    if not folds:
        logger.error("No data splits found.")
        return
    
    latest_fold = folds[-1]
    logger.info(f"Analyzing importance on {latest_fold}")
    
    X_train = pd.read_csv(os.path.join(splits_dir, latest_fold, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(splits_dir, latest_fold, "y_train.csv"))
    
    # y_train might have 'diff', 'home_goals', 'away_goals'. 
    # Use 'diff' as it is the target used for training.
    target = y_train['diff'] if 'diff' in y_train.columns else y_train.iloc[:, 0]
    
    # 2. Train a fast RandomForest to get importance
    # Use only numeric columns, drop metadata
    X_numeric = X_train.select_dtypes(include=[np.number])
    X_numeric = X_numeric.drop(columns=['tournament_year', 'stage_name'], errors='ignore')
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_numeric, target.values.ravel())
    
    # 3. Extract importance
    importances = pd.DataFrame({
        'feature': X_numeric.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    logger.info("Top 20 Features by Importance:")
    print(importances.head(20))
    
    # Save to CSV
    importances.to_csv("models/feature_importance.csv", index=False)
    logger.info("Importance report saved to models/feature_importance.csv")

if __name__ == "__main__":
    analyze_feature_importance()
