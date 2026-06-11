import pandas as pd
import numpy as np
import random
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class TournamentSimulator:
    """
    Simulates matches and tournaments using the trained model and Monte Carlo methods.
    """
    def __init__(self, model):
        self.model = model
        
    def _create_match_features(self, team_a: Dict[str, Any], team_b: Dict[str, Any]) -> pd.DataFrame:
        """
        Creates a feature vector representing a matchup using all 7 engineered features.
        Maps converged prefixed features to core difference features.
        """
        # Map prefixed features from converged data to core expected keys
        # Example mapping: 'elo_ratings_csv__elo' -> 'elo'
        def get_val(team_data, core_key, prefixed_key):
            return float(team_data.get(prefixed_key, team_data.get(core_key, 0.0)))
            
        diff = {
            "diff_elo": get_val(team_a, "elo", "elo_ratings_csv__elo") - get_val(team_b, "elo", "elo_ratings_csv__elo"),
            "diff_ppi": get_val(team_a, "ppi", "conflict_data_csv__intensity") - get_val(team_b, "ppi", "conflict_data_csv__intensity"),
            "diff_uai": get_val(team_a, "uai", "culture_happiness_csv__uai") - get_val(team_b, "uai", "culture_happiness_csv__uai"),
            "diff_ladder": get_val(team_a, "ladder", "happiness_csv__happiness_score") - get_val(team_b, "ladder", "happiness_csv__happiness_score"),
            "diff_clutch": get_val(team_a, "clutch_score_mean", "fifa_world_cup_2026_player_performance_csv__clutch_performance_score") - get_val(team_b, "clutch_score_mean", "fifa_world_cup_2026_player_performance_csv__clutch_performance_score"),
            "diff_tech": get_val(team_a, "technical_floor", "fifa_world_cup_2026_player_performance_csv__performance_score") - get_val(team_b, "technical_floor", "fifa_world_cup_2026_player_performance_csv__performance_score"),
            "diff_pedigree": get_val(team_a, "wc_pedigree", "tournaments_csv__final") - get_val(team_b, "wc_pedigree", "tournaments_csv__final")
        }
                
        return pd.DataFrame([diff])

    def simulate_match(self, team_a: Dict[str, Any], team_b: Dict[str, Any], is_knockout: bool = False) -> Tuple[int, int]:
        """
        Simulates a single match, returning (team_a_goals, team_b_goals).
        """
        X = self._create_match_features(team_a, team_b)
        
        # Predict expected goal difference (Team A goals - Team B goals)
        expected_gd = self.model.predict(X)[0]
        
        # We need expected goals for each team. A simple heuristic:
        # Base goals per team is around 1.2
        expected_goals_a = max(0.1, 1.2 + (expected_gd / 2.0))
        expected_goals_b = max(0.1, 1.2 - (expected_gd / 2.0))
        
        # Poisson distribution for goals
        goals_a = np.random.poisson(expected_goals_a)
        goals_b = np.random.poisson(expected_goals_b)
        
        # Handle knockouts: no draws
        if is_knockout and goals_a == goals_b:
            # Coin flip or slight edge to higher expected goals
            if random.random() < (expected_goals_a / (expected_goals_a + expected_goals_b)):
                goals_a += 1
            else:
                goals_b += 1
                
        return int(goals_a), int(goals_b)
        
    def monte_carlo_match(self, team_a: Dict[str, Any], team_b: Dict[str, Any], iterations: int = 10000) -> Dict[str, float]:
        """
        Simulates a match N times to get probabilities.
        """
        a_wins = 0
        b_wins = 0
        draws = 0
        
        # For efficiency, we shouldn't predict the model 10,000 times, but rather
        # get the expected goals once, then sample Poisson 10,000 times.
        X = self._create_match_features(team_a, team_b)
        expected_gd = self.model.predict(X)[0]
        expected_goals_a = max(0.1, 1.2 + (expected_gd / 2.0))
        expected_goals_b = max(0.1, 1.2 - (expected_gd / 2.0))
        
        samples_a = np.random.poisson(expected_goals_a, iterations)
        samples_b = np.random.poisson(expected_goals_b, iterations)
        
        a_wins = int(np.sum(samples_a > samples_b))
        b_wins = int(np.sum(samples_b > samples_a))
        draws = int(np.sum(samples_a == samples_b))
        
        return {
            "home_win_prob": a_wins / iterations,
            "away_win_prob": b_wins / iterations,
            "draw_prob": draws / iterations,
            "expected_goals_home": float(expected_goals_a),
            "expected_goals_away": float(expected_goals_b)
        }
