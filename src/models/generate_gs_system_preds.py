import pandas as pd
import numpy as np
import pickle
import os
import json
import logging
from src.features.csv_oracle import CSVFeatureOracle
from src.utils.entity_mapper import standardize_country_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_gs_system_predictions():
    # 1. Load the Consensus Oracle
    model_path = "models/v3/Consensus/consensus_model.pkl"
    if not os.path.exists(model_path):
        logger.error(f"Consensus Oracle model not found at {model_path}.")
        return
        
    with open(model_path, "rb") as f:
        consensus_oracle = pickle.load(f)
    
    # 2. Build 2026 Feature Matrix
    oracle_engine = CSVFeatureOracle("data/processed")
    matrix_2026 = oracle_engine.build_2026_matrix()
    
    # Load training feature names for alignment
    with open("models/v3/feature_names.json", "r") as f:
        trained_features = json.load(f)
    
    # 3. Load user matches to predict
    user_preds_path = "models/user_predictions_2026.csv"
    if not os.path.exists(user_preds_path):
        logger.error(f"User predictions file not found at {user_preds_path}.")
        return
    
    user_matches = pd.read_csv(user_preds_path)
    
    system_results = []
    
    logger.info("Generating system predictions using Consensus Oracle...")
    
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
            val = features.get(feat, 0.0)
            feature_dict[feat] = pd.to_numeric(val, errors='coerce')
            
        return pd.Series(feature_dict).fillna(0)

    for _, row in user_matches.iterrows():
        f1 = get_team_data(row['home_team'])
        f2 = get_team_data(row['away_team'])
        
        if f1 is not None and f2 is not None:
            # Prepare Inputs for Consensus
            X1 = f1.to_frame().T
            X2 = f2.to_frame().T
            
            # Use Poisson sub-model goals for consistency: predicted_diff := h_goals - a_goals
            h_goals = consensus_oracle.poisson.home_model.predict(X1)[0]
            a_goals = consensus_oracle.poisson.away_model.predict(X2)[0]
            try:
                score_diff = float(h_goals) - float(a_goals)
            except Exception:
                score_diff = 0.0
            
            # Threshold categorization
            threshold = 0.1
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
                "predicted_home_goals": h_goals,
                "predicted_away_goals": a_goals,
                "predicted_diff": score_diff,
                "predicted_result": result
            })

            
    # Save results
    if system_results:
        system_df = pd.DataFrame(system_results)
    else:
        logger.warning("No predictions generated.")
        system_df = pd.DataFrame(columns=["date", "home_team", "away_team", "predicted_home_goals", "predicted_away_goals", "predicted_diff", "predicted_result"])
        
    # Save as prediction_summary.csv for bootstrap dashboard
    system_df.to_csv("models/prediction_summary.csv", index=False)
    logger.info("System predictions saved to models/system_prediction.csv and models/prediction_summary.csv")

if __name__ == "__main__":
    generate_gs_system_predictions()
