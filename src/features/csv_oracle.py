import pandas as pd
import numpy as np
import os
import logging
import json
from src.features.feature_converger import FeatureConverger
from src.utils.entity_mapper import standardize_country_name
from src.data.entity_resolver import get_iso3_code, resolve_country_name  # fallback resolver

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
        # Surface key converger layers for downstream access/tests
        for attr in ['pol_econ','conflict','elo','fifa23','perf_2026','hofstede','happiness','tournament_context']:
            setattr(self, attr, getattr(self.converger, attr, pd.DataFrame()))

        self.unified_features = pd.read_csv(self.converger.master_features_path)
        with open(self.converger.manifest_path, 'r') as f:
            self.loaded_files = json.load(f)["source_files"]
        
    def get_team_features(self, team_name: str, year: int) -> pd.Series:
        """
        Retrieves all features for a team in a specific year. If the canonical
        lookup fails, attempts several fallbacks and finally builds a proxy by
        aggregating (median) features for the requested year so predictions can
        still be produced for missing teams.
        """
        team_std = standardize_country_name(team_name)
        # If standardizer returns Unknown, try resolver as last resort
        if team_std == 'Unknown':
            team_std = resolve_country_name(team_name)
        
        features = self.unified_features[
            (self.unified_features["canonical_team"] == team_std) & 
            (self.unified_features["year"] == year)
        ]
        
        cols_to_drop = ["canonical_team", "year"]

        if not features.empty:
            row = features.iloc[0]
            if "index" in row.index: cols_to_drop.append("index")
            if "level_0" in row.index: cols_to_drop.append("level_0")
            features_series = row.drop(cols_to_drop, errors='ignore')
            # Ensure numeric types only
            numeric_features = pd.to_numeric(features_series, errors='coerce')
            return numeric_features.fillna(0)

        # Fallbacks - avoid noisy repeated warnings: log only once per missing canonical
        if not hasattr(self, '_logged_missing'):
            self._logged_missing = set()
        if team_std not in getattr(self, '_logged_missing'):
            logger.info(f"Canonical lookup did not find exact match for '{team_name}' (canonical: '{team_std}'). Attempting fallbacks.")
            self._logged_missing.add(team_std)

        # 1) Try ascii-normalized exact or substring matches to canonical_team in same year
        norm = team_std.encode('ascii', 'ignore').decode('ascii').lower()
        candidates = self.unified_features[self.unified_features['year'] == year]['canonical_team'].dropna().unique()

        for c in candidates:
            cnorm = c.encode('ascii', 'ignore').decode('ascii').lower()
            if norm == cnorm:
                row = self.unified_features[(self.unified_features['canonical_team'] == c) & (self.unified_features['year'] == year)].iloc[0]
                if "index" in row.index: cols_to_drop.append("index")
                if "level_0" in row.index: cols_to_drop.append("level_0")
                features_series = row.drop(cols_to_drop, errors='ignore')
                numeric_features = pd.to_numeric(features_series, errors='coerce')
                return numeric_features.fillna(0)

        for c in candidates:
            cnorm = c.encode('ascii', 'ignore').decode('ascii').lower()
            if norm in cnorm or cnorm in norm:
                row = self.unified_features[(self.unified_features['canonical_team'] == c) & (self.unified_features['year'] == year)].iloc[0]
                if "index" in row.index: cols_to_drop.append("index")
                if "level_0" in row.index: cols_to_drop.append("level_0")
                features_series = row.drop(cols_to_drop, errors='ignore')
                numeric_features = pd.to_numeric(features_series, errors='coerce')
                return numeric_features.fillna(0)

        # 2b) Try resolver / ISO3 equivalence to match name variants (e.g., 'Czechia' vs 'Czech Republic')
        try:
            target_iso = get_iso3_code(team_std)
        except Exception:
            target_iso = None

        for c in candidates:
            try:
                # If resolver maps candidate to same canonical, accept it
                if resolve_country_name(c) == team_std:
                    row = self.unified_features[(self.unified_features['canonical_team'] == c) & (self.unified_features['year'] == year)].iloc[0]
                    if "index" in row.index: cols_to_drop.append("index")
                    if "level_0" in row.index: cols_to_drop.append("level_0")
                    features_series = row.drop(cols_to_drop, errors='ignore')
                    numeric_features = pd.to_numeric(features_series, errors='coerce')
                    return numeric_features.fillna(0)

                # Compare ISO3 codes when available
                iso_c = get_iso3_code(c)
                if target_iso and iso_c and iso_c == target_iso:
                    row = self.unified_features[(self.unified_features['canonical_team'] == c) & (self.unified_features['year'] == year)].iloc[0]
                    if "index" in row.index: cols_to_drop.append("index")
                    if "level_0" in row.index: cols_to_drop.append("level_0")
                    features_series = row.drop(cols_to_drop, errors='ignore')
                    numeric_features = pd.to_numeric(features_series, errors='coerce')
                    return numeric_features.fillna(0)
            except Exception:
                # ignore per-candidate failures and continue
                continue

        # 3) As a last resort, return the per-feature median for the requested year
        year_df = self.unified_features[self.unified_features['year'] == year]
        if not year_df.empty:
            # numeric median across year-level teams
            num_df = year_df.select_dtypes(include=['number']).copy()
            if 'year' in num_df.columns:
                num_df = num_df.drop(columns=['year'], errors='ignore')
            med = num_df.median()
            # med is a Series indexed by numeric columns; return with same index order as features normally are
            # Construct a full-feature Series matching the structure of a typical team row
            sample_cols = [c for c in year_df.columns if c not in ('canonical_team','year')]
            med_series = med.reindex(sample_cols, fill_value=0)
            return med_series.fillna(0)

        # If all else fails, return empty series
        return pd.Series(dtype=float)

    def build_absolute_training_set(self):
        """
        Builds X_home, y_home, X_away, y_away, matches from matches.csv
        """
        matches_path = os.path.join(self.processed_dir, "matches.csv")
        if not os.path.exists(matches_path):
            logger.error(f"matches.csv not found at {matches_path}")
            return pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series(), pd.DataFrame()

        matches = pd.read_csv(matches_path)
        x_home_rows = []
        x_away_rows = []
        y_home_vals = []
        y_away_vals = []
        valid_indices = []
        
        logger.info(f"Processing {len(matches)} historical matches for absolute training...")
        
        # Determine column names (handle home_team vs home_team_name)
        home_col = 'home_team' if 'home_team' in matches.columns else 'home_team_name'
        away_col = 'away_team' if 'away_team' in matches.columns else 'away_team_name'
        year_col = 'year' if 'year' in matches.columns else ('tournament_year' if 'tournament_year' in matches.columns else None)
        
        for idx, m in matches.iterrows():
            year = int(m[year_col]) if year_col else 2026
            t1 = m[home_col]
            t2 = m[away_col]
            
            f1 = self.get_team_features(t1, year)
            f2 = self.get_team_features(t2, year)
            
            if f1.empty or f2.empty:
                continue
                
            x_home_rows.append(f1)
            x_away_rows.append(f2)
            y_home_vals.append(m["home_team_score"])
            y_away_vals.append(m["away_team_score"])
            valid_indices.append(idx)
            
        return pd.DataFrame(x_home_rows), pd.Series(y_home_vals), pd.DataFrame(x_away_rows), pd.Series(y_away_vals), matches.loc[valid_indices]

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
