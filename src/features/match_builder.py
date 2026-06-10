import pandas as pd
import numpy as np
from typing import List, Tuple
from sqlalchemy.orm import Session
from src.database import Match, WorldCup
from src.features.matrix_builder import FeatureMatrixBuilder

class MatchTrainingBuilder:
    """
    Converts historical matches and team features into a training set for the Oracle.
    """
    
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.matrix_builder = FeatureMatrixBuilder(session_factory)
        
    def build_training_set(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Creates X (differences) and y (goal differences) from historical matches.
        """
        session = self.session_factory()
        try:
            # 1. Fetch all completed historical matches
            matches = session.query(Match).filter(Match.home_score.isnot(None)).all()
            
            # 2. Get unique tournament years
            years = sorted(list(set(m.tournament_year for m in matches)))
            
            # 3. Cache feature matrices per year to avoid redundant computation
            feature_cache = {}
            for year in years:
                feature_cache[year] = self.matrix_builder.build(year)
                
            X_rows = []
            y_vals = []
            
            for m in matches:
                year_df = feature_cache.get(m.tournament_year)
                if year_df is None or year_df.empty:
                    continue
                    
                team_a_feats = year_df[year_df["team"] == m.home_team]
                team_b_feats = year_df[year_df["team"] == m.away_team]
                
                if team_a_feats.empty or team_b_feats.empty:
                    continue
                
                # Calculate differences (Team A - Team B)
                # Select only numeric columns for difference
                numeric_cols = ["elo", "adversity_mean", "ppi", "legacy_burden", "psyche_score"]
                
                diff = {}
                for col in numeric_cols:
                    if col in team_a_feats.columns and col in team_b_feats.columns:
                        diff[f"diff_{col}"] = team_a_feats[col].values[0] - team_b_feats[col].values[0]
                
                # Add contextual features
                diff["is_host_a"] = 1.0 if team_a_feats["is_host"].values[0] else 0.0
                diff["is_host_b"] = 1.0 if team_b_feats["is_host"].values[0] else 0.0
                
                X_rows.append(diff)
                
                # Target: Goal Difference
                y_vals.append(m.home_score - m.away_score)
                
            X = pd.DataFrame(X_rows)
            y = pd.Series(y_vals)
            
            return X, y
            
        finally:
            session.close()
