import logging
import pandas as pd
import pickle
import os
import json
from src.features.csv_oracle import CSVFeatureOracle
from src.utils.entity_mapper import standardize_country_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_match(team1, team2):
    # Standardize names
    t1_canon = standardize_country_name(team1)
    t2_canon = standardize_country_name(team2)
    
    # 1. Load the Oracle
    model_path = "models/oracle_v1.pkl"
    if not os.path.exists(model_path):
        logger.error(f"Oracle model not found at {model_path}.")
        return
        
    with open(model_path, "rb") as f:
        oracle = pickle.load(f)
    
    # 2. Build 2026 Feature Matrix
    oracle_engine = CSVFeatureOracle("data/processed")
    matrix_2026 = oracle_engine.build_2026_matrix()
    
    # Load training feature names for alignment
    with open("models/feature_names.json", "r") as f:
        trained_features = json.load(f)
    
    def get_team_data(team_name):
        team_df = matrix_2026[matrix_2026["canonical_team"] == team_name]
        if team_df.empty: 
            logger.error(f"Team {team_name} (canonical: {team_name}) not found.")
            return None
        
        # Drop non-feature columns
        features = team_df.iloc[0].drop(['canonical_team', 'year', 'country_code_x', 'country_code_y'], errors='ignore')
        
        # Align to trained_features order and fill missing with 0
        feature_dict = {}
        for feat in trained_features:
            feature_dict[feat.replace("diff_", "")] = features.get(feat.replace("diff_", ""), 0.0)
            
        return pd.Series(feature_dict)

    f1 = get_team_data(t1_canon)
    f2 = get_team_data(t2_canon)
    
    if f1 is not None and f2 is not None:
        # Calculate diffs
        diff = (f1 - f2).add_prefix("diff_")
        
        # Re-order to match trained_features
        X = diff.to_frame().T
        for col in trained_features:
            if col not in X.columns:
                X[col] = 0.0
        X = X[trained_features]
        
        # Predict
        score_diff = oracle.predict(X)[0]
        
        result = "Draw"
        if score_diff > 0: result = f"{team1} ({t1_canon}) wins by {score_diff:.f} goals"
        elif score_diff < 0: result = f"{team2} ({t2_canon}) wins by {abs(score_diff):.f} goals"
        
        logger.info(f"Match Prediction: {team1} vs {team2}")
        logger.info(f"Predicted Goal Difference: {score_diff:.2f}")
        logger.info(f"Result: {result}")
    else:
        logger.error("Could not predict match due to missing team data.")

if __name__ == "__main__":
    predict_match("Canada", "Bosina and Herzegovina")
