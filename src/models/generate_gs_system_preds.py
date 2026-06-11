import pandas as pd
import pickle
import os
import json
import logging
from src.features.csv_oracle import CSVFeatureOracle
from src.utils.entity_mapper import standardize_country_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_system_predictions():
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
    
    # 3. Load user predictions
    user_preds_path = "models/user_predictions_2026.csv"
    if not os.path.exists(user_preds_path):
        logger.error(f"User predictions file not found at {user_preds_path}.")
        return
    
    user_preds = pd.read_csv(user_preds_path)
    
    system_results = []
    
    logger.info("Generating system predictions for all group stage matches...")
    
    def get_team_data(team_name):
        team_canon = standardize_country_name(team_name)
        team_df = matrix_2026[matrix_2026["canonical_team"] == team_canon]
        if team_df.empty: 
            logger.warning(f"Lookup failed for team: '{team_name}' (canonical: '{team_canon}')")
            return None
        
        # Drop non-feature columns
        features = team_df.iloc[0].drop(['canonical_team', 'year', 'country_code_x', 'country_code_y'], errors='ignore')
        
        feature_dict = {}
        for feat in trained_features:
            feature_dict[feat.replace("diff_", "")] = features.get(feat.replace("diff_", ""), 0.0)
            
        return pd.Series(feature_dict)

    for _, row in user_preds.iterrows():
        f1 = get_team_data(row['home_team'])
        f2 = get_team_data(row['away_team'])
        
        if f1 is not None and f2 is not None:
            diff = (f1 - f2).add_prefix("diff_")
            X = diff.to_frame().T
            for col in trained_features:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[trained_features]
            
            score_diff = oracle.predict(X)[0]
            
            # Threshold categorization
            threshold = 0.20
            if score_diff > threshold:
                result = "Win"
            elif score_diff < -threshold:
                result = "Loss"
            else:
                result = "Draw"
            
            system_results.append({
                "date": row['date'],
                "home_team": row['home_team'],
                "away_team": row['away_team'],
                "predicted_diff": score_diff,
                "predicted_result": result
            })
        else:
            logger.warning(f"Could not predict match: {row['home_team']} vs {row['away_team']}")
            
    # Save results
    system_df = pd.DataFrame(system_results)
    system_df.to_csv("models/system_prediction.csv", index=False)
    logger.info("System predictions saved to models/system_prediction.csv")

if __name__ == "__main__":
    generate_system_predictions()
