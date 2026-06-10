import logging
import pandas as pd
import pickle
import os
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
    
    if matrix_2026.empty:
        logger.error("2026 feature matrix is empty. Check data/processed CSV files.")
        return
        
    logger.info(f"Matrix built. Teams: {len(matrix_2026)}")
    
    # 3. Setup Simulator
    simulator = TournamentSimulator(oracle)
    
    # 4. Predict a Sample Match (e.g., USA vs Mexico)
    # Ensure team names match the resolved standardized names
    usa = matrix_2026[matrix_2026["team"] == "United States"].to_dict('records')
    mex = matrix_2026[matrix_2026["team"] == "Mexico"].to_dict('records')
    
    if usa and mex:
        usa = usa[0]
        mex = mex[0]
        probs = simulator.monte_carlo_match(usa, mex)
        logger.info(f"Prediction for USA vs Mexico: {probs}")
    else:
        # If teams not found, just predict the first two
        t1 = matrix_2026.iloc[0].to_dict()
        t2 = matrix_2026.iloc[1].to_dict()
        probs = simulator.monte_carlo_match(t1, t2)
        logger.info(f"Prediction for {t1['team']} vs {t2['team']}: {probs}")

    # 5. Output Win Probabilities (Simple Ranking)
    logger.info("Top 10 Psychopolitical Power Ranking (2026):")
    # Using the columns that exist in CSVFeatureOracle.get_team_features
    print(matrix_2026[["team", "elo", "ppi", "uai", "ladder"]].sort_values("elo", ascending=False).head(10))

if __name__ == "__main__":
    run_2026_prediction()
