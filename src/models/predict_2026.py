import logging
import pandas as pd
import pickle
import os
import json
from src.features.csv_oracle import CSVFeatureOracle
from src.models.simulator import TournamentSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_2026_prediction():
    """
    Predicts the 2026 World Cup outcomes using the trained Oracle and CSV data.
    """
    logger.info("Initializing CSV-based 2026 Prediction Engine...")
    
    # 1. Load the Oracle
    model_path = "models/oracle_v1.pkl"
    if not os.path.exists(model_path):
        logger.error(f"Oracle model not found at {model_path}. Run the training pipeline first.")
        return
        
    with open(model_path, "rb") as f:
        oracle = pickle.load(f)
    
    # 2. Build 2026 Feature Matrix
    oracle_engine = CSVFeatureOracle("data/processed")
    logger.info("Building 2026 Team-Tournament feature matrix from CSVs...")
    matrix_2026 = oracle_engine.build_2026_matrix()
    
    # Load training feature names for alignment
    with open("models/feature_names.json", "r") as f:
        trained_features = json.load(f)
    
    # 3. Setup Simulator
    # We will pass the full matrix to the simulator, not rebuild features
    simulator = TournamentSimulator(oracle)
    
    # 4. Predict a Sample Match (e.g., South Africa vs Mexico)
    # Get team data dicts from the aligned matrix
    def get_team_data(team_name):
        # Use full matrix, not aligned
        team_df = matrix_2026[matrix_2026["canonical_team"] == team_name]
        if team_df.empty: return None
        
        # Ensure we return only the features, dropping metadata
        features = team_df.iloc[0].drop('canonical_team', errors='ignore')
        
        # Align to trained_features order and fill missing with 0
        feature_dict = {}
        for feat in trained_features:
            feature_dict[feat] = features.get(feat, 0.0)
            
        return pd.DataFrame([feature_dict])

    south_africa_data = get_team_data("South Africa")
    mex_data = get_team_data("Mexico")
    
    if south_africa_data is not None and mex_data is not None:
        # Re-calculate differences using identical logic as build_training_set
        # south_africa_data and mex_data are DataFrames
        f1 = south_africa_data.iloc[0]
        f2 = mex_data.iloc[0]
        
        # Difference features: calculate for all prefixed features dynamically
        # Ensure we use the exact same logic as in CSVFeatureOracle
        diff = (f1 - f2).add_prefix("diff_")
        
        # The ensemble expects a 2D DataFrame with column names that match trained_features
        X = diff.to_frame().T
        
        # Ensure X is aligned to trained_features order
        for col in trained_features:
            if col not in X.columns:
                X[col] = 0.0
        X = X[trained_features]
        
        probs = oracle.predict(X)
        logger.info(f"Prediction for South Africa vs Mexico: {probs}")
    else:
        logger.info("Teams not found for prediction.")

    # 5. Output Win Probabilities (Simple Ranking)
    logger.info("Top 10 Psychopolitical Power Ranking (2026):")
    # Using the columns that exist in the converged matrix
    # Mapping old expected names to new prefixed names
    cols = ["canonical_team", "elo_elo", "conflict_intensity", "uai", "happiness_score"]
    # Only select columns that exist in the matrix
    cols_to_print = [c for c in cols if c in matrix_2026.columns]
    print(matrix_2026[cols_to_print].sort_values(cols_to_print[1], ascending=False).head(10))

if __name__ == "__main__":
    run_2026_prediction()
