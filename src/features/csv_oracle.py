import pandas as pd
import numpy as np
import os
import logging
from src.data.entity_resolver import resolve_country_name, get_iso3_code

logger = logging.getLogger(__name__)

class CSVFeatureOracle:
    """
    Unified engine to build feature matrices from CSV files in data/processed.
    Supports both historical training and 2026 prediction.
    """
    
    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = processed_dir
        self._load_base_data()
        
    def _load_base_data(self):
        logger.info("Loading base psychopolitical datasets...")
        self.hofstede = pd.read_csv(os.path.join(self.processed_dir, "hofstede.csv"))
        self.happiness = pd.read_csv(os.path.join(self.processed_dir, "happiness.csv"))
        self.conflict = pd.read_csv(os.path.join(self.processed_dir, "UcdpPrioConflict_v25_1.csv"))
        self.elo_2026 = pd.read_csv(os.path.join(self.processed_dir, "elo.csv"))
        
        # Standardize country names once
        self.hofstede["country"] = self.hofstede["country"].apply(resolve_country_name)
        # Happiness usually has 'country_name' or 'country'
        happy_col = 'country_name' if 'country_name' in self.happiness.columns else 'country'
        self.happiness["country"] = self.happiness[happy_col].apply(resolve_country_name)
        
    def get_team_features(self, team_name: str, year: int) -> dict:
        """
        Retrieves the 11-layer psychopolitical features for a team in a specific year.
        """
        team_std = resolve_country_name(team_name)
        iso3 = get_iso3_code(team_std)
        
        # 1. Cultural (Static)
        hof = self.hofstede[self.hofstede["country"] == team_std]
        uai = hof["uai"].values[0] if not hof.empty else 50.0
        pdi = hof["pdi"].values[0] if not hof.empty else 50.0
        
        # 2. Happiness (National Narrative)
        happy = self.happiness[self.happiness["country"] == team_std]
        # In this CSV, it's called 'happiness_score'
        ladder = happy["happiness_score"].mean() if not happy.empty else 5.0
        
        # 3. Conflict (Political Pressure)
        # Search conflict by sidea or sideb
        side_col = 'sidea' if 'sidea' in self.conflict.columns else 'side_a'
        team_conflict = self.conflict[
            (self.conflict[side_col].str.contains(team_std, na=False)) & 
            (self.conflict["year"] == year)
        ]
        
        # In this CSV, the intensity column name is 'intensity_level'
        intensity_col = 'intensity_level' if 'intensity_level' in self.conflict.columns else 'intensity'
        intensity = team_conflict[intensity_col].max() if not team_conflict.empty else 0.0
        
        # 4. PPI Synthetic
        # Simple version: intensity + low happiness
        ppi = (float(intensity) * 0.7) + ((1.0 - (ladder/10.0)) * 0.3)
        
        # 5. Elo (For 2026 use current, for historical we'd need a time-series Elo)
        # Since we don't have historical Elo CSV yet, we'll use a neutral base if historical
        if year == 2026:
            elo_row = self.elo_2026[self.elo_2026["team"] == team_name]
            elo = elo_row["elo"].values[0] if not elo_row.empty else 1500.0
        else:
            elo = 1500.0 # Placeholder for historical
            
        return {
            "team": team_std,
            "year": year,
            "elo": elo,
            "ppi": ppi,
            "uai": uai,
            "pdi": pdi,
            "ladder": ladder
        }

    def build_training_set(self):
        """
        Builds X, y from matches.csv
        """
        matches = pd.read_csv(os.path.join(self.processed_dir, "matches.csv"))
        X_rows = []
        y_vals = []
        
        logger.info(f"Processing {len(matches)} historical matches...")
        
        for _, m in matches.iterrows():
            year = int(m["tournament_id"].split("-")[1])
            t1 = m["home_team_name"]
            t2 = m["away_team_name"]
            
            f1 = self.get_team_features(t1, year)
            f2 = self.get_team_features(t2, year)
            
            # Difference features
            diff = {
                "diff_elo": f1["elo"] - f2["elo"],
                "diff_ppi": f1["ppi"] - f2["ppi"],
                "diff_uai": f1["uai"] - f2["uai"],
                "diff_ladder": f1["ladder"] - f2["ladder"]
            }
            
            X_rows.append(diff)
            y_vals.append(m["home_team_score"] - m["away_team_score"])
            
        return pd.DataFrame(X_rows), pd.Series(y_vals)

    def build_2026_matrix(self):
        """
        Builds the 2026 contender matrix.
        """
        teams_2026 = pd.read_csv(os.path.join(self.processed_dir, "wc_2026_teams.csv"))
        rows = []
        for _, t in teams_2026.iterrows():
            rows.append(self.get_team_features(t["team"], 2026))
        return pd.DataFrame(rows)
