import logging
import pandas as pd
from src.features.csv_oracle import CSVFeatureOracle
from src.models.ensemble import StackingEnsemble
import pickle
import os

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
    X, y = oracle_engine.build_training_set()
    
    if X.empty:
        logger.error("Training matrix is empty. Check data/processed CSV files.")
        return
        
    logger.info(f"Training set built. Samples: {len(X)}, Features: {list(X.columns)}")
    
    # 2. Train Ensemble
    logger.info("Training Stacking Ensemble (XGBoost + LightGBM + Ridge)...")
    ensemble = StackingEnsemble()
    ensemble.train(X, y)
    
    # 3. Save Model
    os.makedirs("models", exist_ok=True)
    model_path = "models/oracle_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(ensemble, f)
        
    logger.info(f"Oracle model saved successfully to {model_path}")

if __name__ == "__main__":
    run_training_pipeline()
