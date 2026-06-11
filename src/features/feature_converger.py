import pandas as pd
import numpy as np
import logging
import os
import glob
import json
from src.utils.entity_mapper import standardize_country_name

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureConverger:
    def __init__(self, processed_dir="data/processed", master_dir="data/master", models_dir="models"):
        self.processed_dir = processed_dir
        self.master_dir = master_dir
        self.models_dir = models_dir
        self.master_features_path = os.path.join(self.master_dir, 'oracle_master_features.csv')
        self.manifest_path = os.path.join(self.master_dir, 'feature_manifest.json')
        os.makedirs(self.master_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Schema matching heuristics
        self.entity_candidates = ['team', 'team_name', 'country', 'country_code', 'nationality', 'location']

    def _get_entity_column(self, df: pd.DataFrame) -> str:
        """Finds the column representing the team/country."""
        for col in self.entity_candidates:
            if col in df.columns:
                return col
        return None

    def _load_and_standardize(self, filename: str, entity_col: str = None) -> pd.DataFrame:
        """Loads a CSV, finds or uses the entity column, and standardizes it to 'canonical_team'."""
        path = os.path.join(self.processed_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return pd.DataFrame()
            
        logger.info(f"Loading {filename}...")
        df = pd.read_csv(path, low_memory=False)
        
        # Use provided column or heuristic
        if not entity_col:
            entity_col = self._get_entity_column(df)
        
        if not entity_col or entity_col not in df.columns:
            logger.warning(f"Skipping {filename}: No team/country identifier found. Columns: {list(df.columns)}")
            return pd.DataFrame()
            
        # Standardize and drop unresolved
        df['canonical_team'] = df[entity_col].astype(str).apply(standardize_country_name)
        df = df[df['canonical_team'] != 'Unknown'].copy()
        
        # Ensure year exists (default to 2026 if it's static or current data)
        if 'year' not in df.columns and 'tournament_year' not in df.columns:
            df['year'] = 2026
        elif 'tournament_year' in df.columns:
            df['year'] = df['tournament_year']
            
        logger.info(f"Loaded and standardized {filename}. Shape: {df.shape}")
        return df

    def aggregate_player_data(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Aggregates player-level data up to team-year level."""
        if df.empty: return df
        
        logger.info(f"Aggregating player data with prefix '{prefix}'...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['year', 'id', 'jersey_number', 'player_id']]
        
        # Aggregate mean and max (star player factor)
        agg_funcs = {col: ['mean', 'max'] for col in numeric_cols}
        
        team_df = df.groupby(['canonical_team', 'year']).agg(agg_funcs)
        # Flatten multi-level columns
        team_df.columns = [f"{prefix}_{col}_{stat}" for col, stat in team_df.columns]
        
        logger.info(f"Aggregation complete. Shape: {team_df.shape}")
        return team_df.reset_index().copy()

    def _load_tournament_context(self) -> pd.DataFrame:
        """Derives is_host, is_defending_champion, and legacy_weight."""
        path = os.path.join(self.processed_dir, 'world_cups.csv')
        if not os.path.exists(path): return pd.DataFrame()
        
        wc = pd.read_csv(path)
        wc['canonical_host'] = wc['host'].apply(standardize_country_name)
        wc['canonical_winner'] = wc['winner'].apply(standardize_country_name)
        
        context_rows = []
        teams = wc['canonical_host'].unique().tolist() + wc['canonical_winner'].unique().tolist()
        years = range(1930, 2027)
        
        for year in years:
            # Host for this year
            current_wc = wc[wc['year'] == year]
            host = current_wc['canonical_host'].values[0] if not current_wc.empty else None
            
            # Winner of the previous World Cup
            prev_wc = wc[wc['year'] < year].sort_values('year', ascending=False)
            last_winner = prev_wc.iloc[0]['canonical_winner'] if not prev_wc.empty else None
            
            # Cumulative wins before this year
            win_counts = prev_wc['canonical_winner'].value_counts().to_dict() if not prev_wc.empty else {}

            for team in set(teams):
                context_rows.append({
                    'canonical_team': team,
                    'year': year,
                    'is_host': 1 if team == host else 0,
                    'is_defending_champion': 1 if team == last_winner else 0,
                    'legacy_weight': win_counts.get(team, 0)
                })
        
        return pd.DataFrame(context_rows)

    def run(self):
        logger.info("Starting Master Feature Convergence...")

        # 1. Load Match Base
        matches = self._load_and_standardize('matches.csv', entity_col='home_team')
        if matches.empty:
            logger.error("matches.csv is missing or invalid. Cannot build training set.")
            return

        # 2. Load & Aggregate Temporal Datasets
        # Political/Economic
        pol_econ = self._load_and_standardize('political_economic.csv')
        # Conflict
        conflict = self._load_and_standardize('conflict_data.csv')
        if not conflict.empty:
            conflict = conflict.groupby(['canonical_team', 'year']).agg({'intensity': 'max'}).add_prefix('conflict_').reset_index()
            logger.info("Aggregated conflict data.")
        
        # Elo - CRITICAL: Use historical elo
        elo_path = os.path.join(self.processed_dir, 'elo_ratings_historical.csv')
        if os.path.exists(elo_path):
            elo = pd.read_csv(elo_path)
            elo = elo.add_prefix('elo_').rename(columns={'elo_canonical_team': 'canonical_team', 'elo_year': 'year'})
            logger.info("Loaded historical elo data.")
        else:
            elo = pd.DataFrame()

        # 3. Load & Aggregate Player Datasets (Class C - Projections)
        perf = self._load_and_standardize('fifa_world_cup_2026_player_performance.csv')
        team_perf = self.aggregate_player_data(perf, 'perf')

        # 4. Load Static Datasets (Culture)
        culture = self._load_and_standardize('culture_happiness.csv')
        if not culture.empty:
            culture = culture.drop(columns=['year'], errors='ignore').drop_duplicates(subset=['canonical_team'])
            logger.info("Processed culture data.")
            
        # 5. Load Tournament Context (Class B)
        tournament_context = self._load_tournament_context()

        # 6. Build Temporal Spine (1930 - 2026)
        all_teams = sorted(list(set([standardize_country_name(t) for t in pd.concat([matches['home_team'], matches['away_team']]).unique() if t != 'Unknown'])))
        years = list(range(1930, 2027))
        logger.info(f"Constructed temporal spine for {len(all_teams)} teams over {len(years)} years.")
        
        spine = pd.MultiIndex.from_product([all_teams, years], names=['canonical_team', 'year']).to_frame(index=False)

        # 7. Merge Data onto Spine
        logger.info("Merging datasets onto the temporal spine...")
        master = spine.merge(pol_econ, on=['canonical_team', 'year'], how='left')
        master = master.merge(conflict, on=['canonical_team', 'year'], how='left')
        master = master.merge(elo, on=['canonical_team', 'year'], how='left')
        master = master.merge(team_perf, on=['canonical_team', 'year'], how='left')
        master = master.merge(culture, on='canonical_team', how='left')
        master = master.merge(tournament_context, on=['canonical_team', 'year'], how='left')
        logger.info(f"Merge complete. Master DataFrame shape: {master.shape}")

        # 8. FORWARD FILL (Restricted to prevent leakage)
        logger.info("Applying forward fill...")
        master = master.sort_values(['canonical_team', 'year'])
        # Only fill non-temporal/ID columns
        # Class C (perf_) should NOT be forward filled from future to past
        cols_to_fill = [c for c in master.columns if c not in ['canonical_team', 'year'] and not c.startswith('perf_')]
        master[cols_to_fill] = master.groupby('canonical_team')[cols_to_fill].ffill()
        
        # Fill remaining absolute NaNs with 0
        master = master.fillna(0)
        logger.info("Forward fill and NaN filling complete.")

        # Output Master Features
        master.to_csv(self.master_features_path, index=False)
        
        # Save manifest with Class tags
        manifest = {
            "source_files": ["matches.csv", "political_economic.csv", "conflict_data.csv", "elo_ratings_historical.csv", "fifa_world_cup_2026_player_performance.csv", "culture_happiness.csv", "world_cups.csv"],
            "columns": list(master.columns),
            "class_c_features": [c for c in master.columns if c.startswith('perf_')]
        }
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f)

        # 9. Construct the Match Matrix
        logger.info("Constructing Match difference matrix...")
        matches['home_canonical'] = matches['home_team'].apply(standardize_country_name)
        matches['away_canonical'] = matches['away_team'].apply(standardize_country_name)
        
        train_df = matches.copy()
        train_df = train_df.merge(master, left_on=['home_canonical', 'tournament_year'], right_on=['canonical_team', 'year'], how='left')
        train_df = train_df.merge(master, left_on=['away_canonical', 'tournament_year'], right_on=['canonical_team', 'year'], how='left', suffixes=('_home', '_away'))

        # DROP CLASS C FOR TRAINING (Firewall)
        class_c = manifest["class_c_features"]
        feature_cols = [c for c in master.columns if c not in ['canonical_team', 'year', 'country_code_x', 'country_code_y'] and c not in class_c]
        
        diff_data = {}
        for col in feature_cols:
            h_col, a_col = f"{col}_home", f"{col}_away"
            if h_col in train_df.columns and a_col in train_df.columns:
                diff_data[f"diff_{col}"] = pd.to_numeric(train_df[h_col], errors='coerce').fillna(0) - pd.to_numeric(train_df[a_col], errors='coerce').fillna(0)
        
        diff_df = pd.DataFrame(diff_data, index=train_df.index)
        y = pd.to_numeric(train_df['home_team_score'], errors='coerce').fillna(0) - pd.to_numeric(train_df['away_team_score'], errors='coerce').fillna(0)
        
        final_train_df = pd.concat([diff_df, y.rename('y')], axis=1)
        final_train_df.to_csv(os.path.join(self.models_dir, 'train.csv'), index=False)
        logger.info(f"Training matrix built. Shape: {final_train_df.shape}")
