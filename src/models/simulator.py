import pandas as pd
import numpy as np
import random
import logging
import os
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class TournamentSimulator:
    """
    Simulates matches and tournaments using the trained ConsensusOracle and Monte Carlo methods.
    """
    def __init__(self, model):
        self.model = model
        self._valid_features = None
        
        # Load valid features once to avoid redundant I/O
        feature_path = "models/v3/feature_names.json"
        if os.path.exists(feature_path):
            import json
            with open(feature_path, "r") as f:
                self._valid_features = json.load(f)
        
    def _prepare_team_features(self, team_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Converts team data dictionary to the feature DataFrame expected by the Oracle.
        Uses the model's trained feature list when possible to guarantee alignment and numeric types.
        """
        # Determine canonical feature list to use, prefer model's features when available
        feature_list = None
        try:
            # If ConsensusOracle with poisson submodels, prefer away_model's features
            if hasattr(self.model, 'poisson'):
                away = getattr(self.model.poisson, 'away_model', None)
                home = getattr(self.model.poisson, 'home_model', None)
                # pick away model feature columns if present
                if away and hasattr(away, '_feature_columns'):
                    feature_list = list(away._feature_columns)
                elif home and hasattr(home, '_feature_columns'):
                    feature_list = list(home._feature_columns)
            # Generic model may expose _feature_columns or feature_names
            if feature_list is None and hasattr(self.model, '_feature_columns'):
                feature_list = list(self.model._feature_columns)
            if feature_list is None and hasattr(self.model, 'feature_names'):
                feature_list = list(self.model.feature_names)
        except Exception:
            feature_list = None

        # Fall back to packaged feature file
        if feature_list is None:
            if self._valid_features is None:
                import json
                with open("models/v3/feature_names.json", "r") as f:
                    self._valid_features = json.load(f)
            feature_list = list(self._valid_features)

        # Helper to safely convert to float
        def safe_float(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        # Build dict using only numeric features that the model expects
        feature_dict = {}
        for feat in feature_list:
            # Skip known non-numeric id fields
            if feat in ('country_code', 'tournament_year'):
                # attempt to coerce tournament_year to numeric if present
                if feat == 'tournament_year':
                    feature_dict[feat] = safe_float(team_data.get(feat, 0.0))
                else:
                    # for country codes, set 0
                    feature_dict[feat] = 0.0
                continue
            feature_dict[feat] = safe_float(team_data.get(feat, 0.0))

        # Ensure deterministic column order matching feature_list
        df = pd.DataFrame([feature_dict])[feature_list]
        return df

    def simulate_match(self, team_a_data: Dict[str, Any], team_b_data: Dict[str, Any], is_knockout: bool = False) -> Tuple[int, int]:
        """
        Simulates a single match using ConsensusOracle, returning (team_a_goals, team_b_goals).
        """
        # If the input is already a DataFrame, use it directly (optimization for pre-calc)
        if isinstance(team_a_data, pd.DataFrame):
            X1 = team_a_data
        else:
            X1 = self._prepare_team_features(team_a_data)
            
        if isinstance(team_b_data, pd.DataFrame):
            X2 = team_b_data
        else:
            X2 = self._prepare_team_features(team_b_data)
        
        # Predict expected goals using Poisson sub-models or generic model.predict
        if hasattr(self.model, 'poisson'):
            expected_goals_a = max(0.1, float(self.model.poisson.home_model.predict(X1)[0]))
            expected_goals_b = max(0.1, float(self.model.poisson.away_model.predict(X2)[0]))
        elif hasattr(self.model, 'predict'):
            expected_goals_a = max(0.1, float(self.model.predict(X1)[0]))
            expected_goals_b = max(0.1, float(self.model.predict(X2)[0]))
        else:
            expected_goals_a, expected_goals_b = 0.1, 0.1
        
        # Poisson distribution for goals
        goals_a = np.random.poisson(expected_goals_a)
        goals_b = np.random.poisson(expected_goals_b)
        
        # Handle knockouts: no draws
        if is_knockout and goals_a == goals_b:
            if random.random() < (expected_goals_a / (expected_goals_a + expected_goals_b)):
                goals_a += 1
            else:
                goals_b += 1
                
        return int(goals_a), int(goals_b)
        
    def monte_carlo_match(self, team_a_data: Dict[str, Any], team_b_data: Dict[str, Any], iterations: int = 25000) -> Dict[str, float]:
        """
        Simulates a match N times to get probabilities using ConsensusOracle Poisson sub-models.
        """
        X1 = self._prepare_team_features(team_a_data)
        X2 = self._prepare_team_features(team_b_data)
        
        expected_goals_a = max(0.1, float(self.model.poisson.home_model.predict(X1)[0]))
        expected_goals_b = max(0.1, float(self.model.poisson.away_model.predict(X2)[0]))
        
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
