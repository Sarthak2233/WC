import pickle
import logging
import argparse
import pandas as pd
from src.models.simulator import TournamentSimulator
from src.models.full_tournament_sim import FullTournamentSimulator, _load_engine_from_disk
from src.features.csv_oracle import CSVFeatureOracle
from src.utils.entity_mapper import standardize_country_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_simulation(full_tournament: bool = False):
    if full_tournament:
        logger.info("Starting Full Tournament Monte Carlo Simulation...")
        from src.models.full_tournament_sim import run_full_tournament_mc
        run_full_tournament_mc(n_iterations=10000)
        logger.info("Full Tournament Simulations complete. Results saved in models/mc_win_probabilities.csv")
        return

    logger.info("Starting Monte Carlo Simulation for all matches...")
    model_path = "models/v3/Consensus/consensus_model.pkl"
    with open(model_path, "rb") as f:
        oracle = pickle.load(f)
    
    simulator = TournamentSimulator(oracle)
    
    oracle_engine = CSVFeatureOracle("data/processed")
    matrix_2026 = oracle_engine.build_2026_matrix()
    
    # Load all matches to simulate
    user_preds_path = "models/user_predictions_2026.csv"
    user_matches = pd.read_csv(user_preds_path)
    
    simulation_results = []
    
    for _, row in user_matches.iterrows():
        t1_canon = standardize_country_name(row['home_team'])
        t2_canon = standardize_country_name(row['away_team'])
        
        team_a_df = matrix_2026[matrix_2026["canonical_team"] == t1_canon]
        team_b_df = matrix_2026[matrix_2026["canonical_team"] == t2_canon]
        
        if team_a_df.empty or team_b_df.empty:
            logger.warning(f"Could not find data for match: {row['home_team']} vs {row['away_team']}")
            continue
            
        team_a = team_a_df.iloc[0].to_dict()
        team_b = team_b_df.iloc[0].to_dict()
        
        probs = simulator.monte_carlo_match(team_a, team_b)
        
        probs['home_team'] = row['home_team']
        probs['away_team'] = row['away_team']
        simulation_results.append(probs)
    
    results_df = pd.DataFrame(simulation_results)
    results_df.to_csv("models/simulation_results.csv", index=False)
    logger.info("Monte Carlo Simulations complete. Results saved to models/simulation_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tournament simulations.")
    parser.add_argument("--full-tournament", action="store_true", help="Run full 10,000-run tournament simulation.")
    args = parser.parse_args()
    run_simulation(full_tournament=args.full_tournament)
