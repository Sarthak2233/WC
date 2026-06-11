import logging
import pandas as pd
from src.features.csv_oracle import CSVFeatureOracle
from src.models.ensemble import StackingEnsemble
import pickle
import os
import json
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training_pipeline():
    """
    Orchestrates the training of the Psychopolitical Oracle using CSV data.
    """
    logger.info("Initializing CSV-based Oracle Training Pipeline...")
    
    oracle_engine = CSVFeatureOracle("data/processed")
    
    # 1. Build Training Set
    logger.info("Building historical training matrix from CSVs...")
    X, y, teams = oracle_engine.build_training_set()

    if X.empty:
        logger.error("Training matrix is empty. Check data/processed CSV files.")
        return

    # Phase 2: Deterministic Split
    logger.info("Performing deterministic train/test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get dynamic source list
    source_list = ",".join(oracle_engine.loaded_files)

    # Add metadata for auditability
    X_train_export = X_train.copy()
    X_train_export['y'] = y_train
    X_train_export['SOURCE_FILES'] = source_list

    X_test_export = X_test.copy()
    X_test_export['y'] = y_test
    X_test_export['SOURCE_FILES'] = source_list

    os.makedirs("models", exist_ok=True)
    X_train_export.to_csv("models/train.csv", index=False)
    X_test_export.to_csv("models/test.csv", index=False)
    logger.info("Exported train.csv and test.csv to models/")

    # Drop metadata from training features
    X_train_clean = X_train.drop(columns=['y', 'SOURCE_FILES'], errors='ignore')

    logger.info(f"Training set built. Samples: {len(X)}, Features: {list(X_train_clean.columns)}")

    # 2. Train Ensemble
    logger.info("Training Stacking Ensemble (XGBoost + LightGBM + Ridge)...")
    ensemble = StackingEnsemble()
    ensemble.train(X_train_clean, y_train)

    # 3. Save Model & Metadata
    os.makedirs("models", exist_ok=True)
    model_path = "models/oracle_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(ensemble, f)

    # Save feature names for alignment
    with open("models/feature_names.json", "w") as f:
        json.dump(X_train_clean.columns.tolist(), f)

    # Save teams metadata for auditability
    teams.to_csv("models/training_teams.csv", index=False)

    logger.info(f"Oracle model saved successfully to {model_path}")
    logger.info("Feature names saved to models/feature_names.json")


if __name__ == "__main__":
    run_training_pipeline()
