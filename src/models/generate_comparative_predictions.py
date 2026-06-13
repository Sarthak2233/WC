import pandas as pd
import pickle
import os
import json
import logging
from src.features.csv_oracle import CSVFeatureOracle
from src.utils.entity_mapper import standardize_country_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_comparative_report():
    base_path = "models/v3"
    # Load all available models from v3
    models = {
        "Ensemble": f"{base_path}/Ensemble/ensemble_model.pkl",
        "Bayesian": f"{base_path}/Bayesian/bayesianhierarchicaltrainer_model.pkl",
        "HomePoisson": f"{base_path}/HomePoisson/homepoisson_model.pkl",
        "AwayPoisson": f"{base_path}/AwayPoisson/awaypoisson_model.pkl",
        "Consensus": f"{base_path}/Consensus/consensus_model.pkl"
    }
    
    loaded_models = {}
    for name, path in models.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                loaded_models[name] = pickle.load(f)
        else:
            logger.warning(f"Model not found: {path}")

    # Build 2026 Feature Matrix
    oracle_engine = CSVFeatureOracle("data/processed")
    matrix_2026 = oracle_engine.build_2026_matrix()
    
    # Load training feature names
    with open(f"{base_path}/feature_names.json", "r") as f:
        trained_features = json.load(f)
    
    # Load user matches
    user_preds_path = "models/user_predictions_2026.csv"
    user_matches = pd.read_csv(user_preds_path)
    
    results = []
    
    logger.info("Generating comparative predictions...")
    
    def get_team_data(team_name, features_list):
        team_canon = standardize_country_name(team_name)
        team_df = matrix_2026[matrix_2026["canonical_team"] == team_canon]
        if team_df.empty: return None
        
        # Get absolute features
        features = team_df.iloc[0].drop(['canonical_team', 'year', 'country_code_x', 'country_code_y'], errors='ignore')
        
        feature_dict = {}
        for feat in features_list:
            val = features.get(feat, 0.0)
            feature_dict[feat] = pd.to_numeric(val, errors='coerce')
        return pd.Series(feature_dict).fillna(0)

    for _, row in user_matches.iterrows():
        f1 = get_team_data(row['home_team'], trained_features)
        f2 = get_team_data(row['away_team'], trained_features)
        
        if f1 is None or f2 is None: continue
        
        # Prepare Inputs
        X1 = f1.to_frame().T
        X2 = f2.to_frame().T
        X_diff = (f1 - f2).to_frame().T
        X_diff.columns = [f"diff_{c}" for c in X_diff.columns]
        
        match_result = {
            "date": row['date'],
            "home_team": row['home_team'],
            "away_team": row['away_team']
        }
        
        # Generate predictions for each model
        # 1. Ensemble (Difference model)
        if "Ensemble" in loaded_models:
            match_result["Ensemble_diff"] = loaded_models["Ensemble"].predict(X_diff)[0]
            
        # 2. Bayesian (Difference model)
        if "Bayesian" in loaded_models:
            # Bayesian might use stage_name
            X_b = X_diff.copy()
            X_b['stage_name'] = row['stage']
            match_result["Bayesian_diff"] = loaded_models["Bayesian"].predict(X_b)[0]
            
        # 3. Poisson (Absolute models)
        if "HomePoisson" in loaded_models and "AwayPoisson" in loaded_models:
            match_result["Poisson_home"] = loaded_models["HomePoisson"].predict(X1)[0]
            match_result["Poisson_away"] = loaded_models["AwayPoisson"].predict(X2)[0]
            match_result["Poisson_diff"] = match_result["Poisson_home"] - match_result["Poisson_away"]
            
        # 4. Consensus
        if "Consensus" in loaded_models:
            match_result["Consensus_diff"] = loaded_models["Consensus"].predict_match(X1, X2)[0]
            
        results.append(match_result)

            
    # Save report
    pd.DataFrame(results).to_csv("models/comparative_prediction_report.csv", index=False)
    logger.info("Comparative report saved to models/comparative_prediction_report.csv")

if __name__ == "__main__":
    generate_comparative_report()
