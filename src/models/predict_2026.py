import logging
import pandas as pd
import pickle
import os
import json
import sys
from src.features.csv_oracle import CSVFeatureOracle
from src.utils.entity_mapper import standardize_country_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_2026_prediction(team1=None, team2=None):
    """
    Predicts outcomes for 2026. If team1/team2 provided, predicts specific match.
    """
    logger.info("Initializing CSV-based 2026 Prediction Engine...")
    
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
    
    if team1 and team2:
        t1_canon = standardize_country_name(team1)
        t2_canon = standardize_country_name(team2)
        
        def get_team_data(team_name):
            team_df = matrix_2026[matrix_2026["canonical_team"] == team_name]
            if team_df.empty: 
                logger.error(f"Team {team_name} (canonical: {team_name}) not found.")
                return None
            features = team_df.iloc[0].drop(['canonical_team', 'year', 'country_code_x', 'country_code_y'], errors='ignore')
            feature_dict = {}
            for feat in trained_features:
                feature_dict[feat.replace("diff_", "")] = features.get(feat.replace("diff_", ""), 0.0)
            return pd.Series(feature_dict)

        f1 = get_team_data(t1_canon)
        f2 = get_team_data(t2_canon)
        
        if f1 is not None and f2 is not None:
            diff = (f1 - f2).add_prefix("diff_")
            X = diff.to_frame().T
            for col in trained_features:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[trained_features]
            
            score_diff = oracle.predict(X)[0]
            
            # Apply threshold for categorical interpretation
            threshold = 0.20
            if score_diff > threshold:
                result = f"{team1} ({t1_canon}) wins"
            elif score_diff < -threshold:
                result = f"{team2} ({t2_canon}) wins"
            else:
                result = "Draw"
            
            logger.info(f"Match Prediction: {team1} vs {team2}")
            logger.info(f"Predicted Goal Difference: {score_diff:.2f}")
            logger.info(f"Result: {result}")
        else:
            logger.error("Could not predict match.")
    else:
        # Default behavior: Print rankings
        logger.info("Top 10 Psychopolitical Power Ranking (2026):")
        cols = ["canonical_team", "elo_elo", "conflict_intensity", "uai", "happiness_score"]
        cols_to_print = [c for c in cols if c in matrix_2026.columns]
        print(matrix_2026[cols_to_print].sort_values(cols_to_print[1], ascending=False).head(10))

if __name__ == "__main__":
    if len(sys.argv) > 2:
        run_2026_prediction(sys.argv[1], sys.argv[2])
    else:
        run_2026_prediction()
