import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from src.features.csv_oracle import CSVFeatureOracle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_baseline():
    """
    Evaluates a simple Elo-only baseline against historical data.
    """
    engine = CSVFeatureOracle("data/processed")
    X, y_actual, _ = engine.build_training_set()
    # Baseline only uses diff_elo_elo
    X_baseline = X[["diff_elo_elo"]]
    
    # Drop rows with NaN
    mask = ~X_baseline.isna().any(axis=1)
    X_baseline = X_baseline[mask]
    y_actual = y_actual[mask]
    
    # Train simple linear regressor
    baseline_model = LinearRegression()
    baseline_model.fit(X_baseline, y_actual)
    y_pred = baseline_model.predict(X_baseline)
    
    # Metrics
    mae = mean_absolute_error(y_actual, y_pred)
    
    def to_result(score):
        if score > 0: return 1
        if score < 0: return -1
        return 0
        
    actual_results = y_actual.apply(to_result)
    pred_results = pd.Series(y_pred).apply(to_result)
    accuracy = accuracy_score(actual_results, pred_results)
    
    logger.info("--- Baseline (Elo-Only) Performance Statistics ---")
    logger.info(f"Mean Absolute Error (Goal Diff): {mae:.4f}")
    logger.info(f"Match Result Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate_baseline()
