import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn.metrics import accuracy_score, mean_absolute_error, log_loss
from src.features.csv_oracle import CSVFeatureOracle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model():
    """
    Evaluates the trained Oracle model against historical data.
    """
    model_path = "models/oracle_v1.pkl"
    if not os.path.exists(model_path):
        logger.error(f"Oracle model not found at {model_path}.")
        return
        
    with open(model_path, "rb") as f:
        oracle = pickle.load(f)
        
    engine = CSVFeatureOracle("data/processed")
    X, y_actual, _ = engine.build_training_set()
    
    # Generate predictions
    y_pred = oracle.predict(X)
    
    # 1. Regression Metrics (Goal Difference)
    mae = mean_absolute_error(y_actual, y_pred)
    
    # 2. Classification Metrics (Win/Loss/Draw)
    def to_result(score):
        if score > 0: return 1 # Win
        if score < 0: return -1 # Loss
        return 0 # Draw
        
    actual_results = y_actual.apply(to_result)
    pred_results = pd.Series(y_pred).apply(to_result)
    
    accuracy = accuracy_score(actual_results, pred_results)
    
    logger.info("--- Model Performance Statistics ---")
    logger.info(f"Mean Absolute Error (Goal Diff): {mae:.4f}")
    logger.info(f"Match Result Accuracy: {accuracy:.4f}")
    
    # For LogLoss, we need probability distributions. 
    # Since this is a regressor, we can approximate probabilities using the simulator's logic, 
    # but for now, this basic accuracy gives a strong baseline.

if __name__ == "__main__":
    evaluate_model()
