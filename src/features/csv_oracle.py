import pandas as pd
import numpy as np
import os
import logging
import json
from src.features.feature_converger import FeatureConverger
from src.data.entity_resolver import resolve_country_name

logger = logging.getLogger(__name__)

class CSVFeatureOracle:
    """
    Unified engine to build feature matrices from CSV files in data/processed.
    Supports both historical training and 2026 prediction.
    """
    
    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = processed_dir
        self.converger = FeatureConverger(processed_dir)
        self.converger.run()
        self.unified_features = pd.read_csv(self.converger.master_features_path)
        with open(self.converger.manifest_path, 'r') as f:
            self.loaded_files = json.load(f)["source_files"]
        
    def get_team_features(self, team_name: str, year: int) -> pd.Series:
        """
        Retrieves all features for a team in a specific year.
        """
        team_std = resolve_country_name(team_name)
        
        features = self.unified_features[
            (self.unified_features["canonical_team"] == team_std) & 
            (self.unified_features["year"] == year)
        ]
        
        if features.empty:
            return pd.Series(dtype=float)
            
        row = features.iloc[0]
        cols_to_drop = ["canonical_team", "year"]
        if "index" in row.index: cols_to_drop.append("index")
        if "level_0" in row.index: cols_to_drop.append("level_0")
        
        features_series = row.drop(cols_to_drop, errors="ignore")
        
        # Ensure numeric types only
        numeric_features = pd.to_numeric(features_series, errors='coerce')
        
        return numeric_features

    def build_training_set(self):
        """
        Builds X, y from matches.csv
        """
        matches = pd.read_csv(os.path.join(self.processed_dir, "matches.csv"))
        x_rows = []
        y_vals = []
        team_rows = []
        
        logger.info(f"Processing {len(matches)} historical matches...")
        
        for _, m in matches.iterrows():
            year = int(m["tournament_year"])
            t1 = m["home_team"]
            t2 = m["away_team"]
            
            f1 = self.get_team_features(t1, year)
            f2 = self.get_team_features(t2, year)
            
            # Difference features: calculate for all prefixed features dynamically
            diff = (f1 - f2).add_prefix("diff_")
            
            x_rows.append(diff)
            y_vals.append(m["home_team_score"] - m["away_team_score"])
            team_rows.append({'home_team': t1, 'away_team': t2})
            
        return pd.DataFrame(x_rows), pd.Series(y_vals), pd.DataFrame(team_rows)

    def build_2026_matrix(self):
        """
        Builds the 2026 contender matrix with all features.
        """
        # Start with the unified features specifically for the year 2026
        features_2026 = self.unified_features[self.unified_features['year'] == 2026].copy()

        # Ensure the 'canonical_team' column is present.
        if 'canonical_team' not in features_2026.columns:
            logger.error("Unified features missing 'canonical_team' column.")
            return pd.DataFrame()

        return features_2026
