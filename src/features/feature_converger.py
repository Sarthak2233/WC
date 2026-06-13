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
        for col in self.entity_candidates + ['home_team', 'home_team_name']:
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
        
        # Use provided column or fallback to heuristic
        final_col = entity_col if entity_col and entity_col in df.columns else self._get_entity_column(df)
        
        if not final_col:
            logger.warning(f"Skipping {filename}: No team/country identifier found. Columns: {list(df.columns)}")
            return pd.DataFrame()
            
        # Standardize and drop unresolved
        df['canonical_team'] = df[final_col].astype(str).apply(standardize_country_name)
        
        # Ensure year exists (Class B/C Logic)
        if 'year' not in df.columns and 'tournament_year' not in df.columns:
            if 'tournament_id' in df.columns:
                try:
                    df['tournament_year'] = df['tournament_id'].apply(lambda x: int(str(x).split("-")[1]))
                    df['year'] = df['tournament_year']
                except:
                    df['year'] = 2026
            else:
                df['year'] = 2026
        elif 'tournament_year' in df.columns and 'year' not in df.columns:
            df['year'] = df['tournament_year']
        elif 'year' in df.columns and 'tournament_year' not in df.columns:
            df['tournament_year'] = df['year']
            
        # Audit resolution
        total_rows = len(df)
        resolved_rows = len(df[df['canonical_team'] != 'Unknown'])
        resolution_rate = (resolved_rows / total_rows) * 100 if total_rows > 0 else 0
        logger.info(f"Resolution Audit for {filename}: {resolution_rate:.2f}% ({resolved_rows}/{total_rows})")
            
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
        
        # Flatten columns more efficiently
        team_df.columns = [f"{prefix}_{col}_{stat}" for col, stat in team_df.columns]
        
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
        logger.info("Starting Master Feature Convergence with Strict Temporal Alignment...")

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
        
        # Elo - Use historical elo
        elo_path = os.path.join(self.processed_dir, 'elo_ratings_historical.csv')
        elo = pd.DataFrame()
        if os.path.exists(elo_path):
            elo = pd.read_csv(elo_path)
            elo = elo.add_prefix('elo_').rename(columns={'elo_canonical_team': 'canonical_team', 'elo_year': 'year'})
            logger.info("Loaded historical elo data.")

        # 3. Load Player Data
        # Historical Anchor (FIFA 23 for 2022)
        fifa23_path = os.path.join(self.processed_dir, 'players_fifa23.csv')
        fifa23_data = pd.DataFrame()
        if os.path.exists(fifa23_path):
            df_f23 = pd.read_csv(fifa23_path, low_memory=False)
            df_f23['canonical_team'] = df_f23['country'].apply(standardize_country_name)
            df_f23['year'] = 2022
            cols = ['overall', 'potential', 'pace_total', 'shooting_total', 'passing_total', 'dribbling_total', 'defending_total', 'physicality_total']
            fifa23_data = df_f23.groupby(['canonical_team', 'year'])[cols].mean().add_prefix('perf_').reset_index()
            logger.info("Aggregated FIFA 23 data as 2022 anchor.")

        # 2026 Projections (Isolated)
        perf_2026_path = os.path.join(self.processed_dir, '2026_only', 'fifa_world_cup_2026_player_performance.csv')
        perf_2026_data = pd.DataFrame()
        if os.path.exists(perf_2026_path):
            df_p26 = pd.read_csv(perf_2026_path, low_memory=False)
            # Find team column
            team_col = 'team' if 'team' in df_p26.columns else 'canonical_team'
            df_p26['canonical_team'] = df_p26[team_col].apply(standardize_country_name)
            df_p26['year'] = 2026
            perf_2026_data = self.aggregate_player_data(df_p26, 'perf')
            logger.info("Aggregated 2026 performance projections.")

        # 4. Load Static Datasets (Culture)
        culture = self._load_and_standardize('culture_happiness.csv')
        if not culture.empty:
            # Separate static Hofstede from temporal Happiness
            hofstede_cols = [c for c in culture.columns if 'hofstede' in c or c == 'canonical_team']
            hofstede = culture[hofstede_cols].drop_duplicates(subset=['canonical_team'])
            
            happiness = culture[['canonical_team', 'year', 'happiness_score']].copy()
            logger.info("Separated static Hofstede and temporal Happiness.")
        else:
            hofstede = pd.DataFrame()
            happiness = pd.DataFrame()

        # 5. Load Tournament Context
        tournament_context = self._load_tournament_context()

        # Persist key loaded layers for external access/tests
        self.pol_econ = pol_econ
        self.conflict = conflict
        self.elo = elo
        self.fifa23 = fifa23_data
        self.perf_2026 = perf_2026_data
        self.hofstede = hofstede if 'hofstede' in locals() else pd.DataFrame()
        self.happiness = happiness if 'happiness' in locals() else pd.DataFrame()
        self.tournament_context = tournament_context

        # 6. Build Temporal Spine (1930 - 2026)
        historical_teams = set([standardize_country_name(t) for t in pd.concat([matches['home_team'], matches['away_team']]).unique() if t != 'Unknown'])
        
        # Add 2026 contenders from source of truth
        contenders_2026_path = os.path.join(self.processed_dir, 'wc_2026_teams.csv')
        if os.path.exists(contenders_2026_path):
            contenders_2026 = pd.read_csv(contenders_2026_path)
            c26_teams = set(contenders_2026['team'].apply(standardize_country_name).unique())
            historical_teams.update(c26_teams)
            logger.info(f"Added {len(c26_teams)} 2026 contenders to temporal spine.")
            
        all_teams = sorted(list(historical_teams))
        years = list(range(1930, 2027))
        spine = pd.MultiIndex.from_product([all_teams, years], names=['canonical_team', 'year']).to_frame(index=False)

        # 7. Merge Data onto Spine
        logger.info("Merging datasets onto the temporal spine...")
        master = spine.merge(pol_econ, on=['canonical_team', 'year'], how='left')
        master = master.merge(conflict, on=['canonical_team', 'year'], how='left')
        master = master.merge(elo, on=['canonical_team', 'year'], how='left')
        
        # Combine performance data
        combined_perf = pd.concat([fifa23_data, perf_2026_data], ignore_index=True)
        master = master.merge(combined_perf, on=['canonical_team', 'year'], how='left')
        
        master = master.merge(happiness, on=['canonical_team', 'year'], how='left')
        master = master.merge(hofstede, on='canonical_team', how='left')
        master = master.merge(tournament_context, on=['canonical_team', 'year'], how='left')

        # Defragment to avoid PerformanceWarning
        master = master.copy()

        # Normalize canonical_team naming to the project's standard to avoid name mismatches
        try:
            master['canonical_team'] = master['canonical_team'].astype(str).apply(standardize_country_name)
        except Exception:
            logger.exception('Failed to normalize canonical_team names; continuing without normalization.')

        # Ensure all 2026 contenders exist in master (inject available 2026 squad/perf values)
        if os.path.exists(contenders_2026_path):
            try:
                contenders_2026 = pd.read_csv(contenders_2026_path)
                contenders_2026['canonical_team'] = contenders_2026['team'].apply(standardize_country_name)
                c26_teams = set([t for t in contenders_2026['canonical_team'].unique() if t and t != 'Unknown'])
                master_2026_teams = set(master[master['year'] == 2026]['canonical_team'].unique())
                missing_teams = c26_teams - master_2026_teams
                if missing_teams:
                    logger.info(f"Injecting {len(missing_teams)} missing 2026 contenders into master: {sorted(list(missing_teams))}")
                    rows = []
                    for t in missing_teams:
                        # zero-default row
                        row = {c: 0 for c in master.columns}
                        row['canonical_team'] = t
                        row['year'] = 2026
                        # inject performance/squad features from combined_perf when available
                        cp = combined_perf[(combined_perf['canonical_team'] == t) & (combined_perf['year'] == 2026)]
                        if not cp.empty:
                            for col in cp.columns:
                                if col in master.columns and col not in ('canonical_team', 'year'):
                                    row[col] = cp.iloc[0].get(col, 0)
                        rows.append(row)
                    if rows:
                        master = pd.concat([master, pd.DataFrame(rows)], ignore_index=True)
            except Exception:
                logger.exception('Error while ensuring 2026 contenders present in master. Continuing.')


        # 8. STRICT TEMPORAL FILL (FFILL ONLY)
        logger.info("Performing strict temporal forward-fill...")
        master = master.sort_values(['canonical_team', 'year'])
        cols_to_ffill = [c for c in master.columns if c not in ['canonical_team', 'year', 'confederation', 'stage_name']]
        master[cols_to_ffill] = master.groupby('canonical_team')[cols_to_ffill].ffill()
        
        # Fill remaining with 0
        master = master.fillna(0)
        logger.info("Temporal fill complete.")
        
        # 8.b Feature variance and zero-rate filtering to remove near-constant columns
        feature_drop = os.environ.get('FEATURE_DROP', 'true').lower() in ('1', 'true', 'yes')
        numeric_cols = master.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            zero_rates = ((master[numeric_cols] == 0).sum() / len(master)).to_dict()
            variances = master[numeric_cols].var().to_dict()
        else:
            zero_rates = {}
            variances = {}
        # Determine features to drop (high zero_rate or near-zero variance)
        drop_threshold = 0.95
        var_threshold = 1e-6
        whitelist_prefixes = ['elo_', 'is_host', 'legacy_weight', 'happiness_score']
        filtered = []
        for c in numeric_cols:
            zr = float(zero_rates.get(c, 0.0))
            var = float(variances.get(c, 0.0))
            if (zr > drop_threshold) or (var < var_threshold):
                if not any(c.startswith(p) for p in whitelist_prefixes):
                    filtered.append(c)
        if filtered and feature_drop:
            logger.info(f"Dropping {len(filtered)} near-constant features from master: {filtered[:10]}{'...' if len(filtered)>10 else ''}")
            master = master.drop(columns=filtered)
        elif filtered and not feature_drop:
            logger.info(f"FEATURE_DROP disabled: keeping near-constant features: {filtered[:10]}{'...' if len(filtered)>10 else ''}")
        else:
            logger.info('No near-constant features dropped.')
        # Save diagnostic variance report
        var_report_path = os.path.join(self.master_dir, 'feature_variance_report.csv')
        with open(var_report_path, 'w') as rf:
            rf.write('feature,zero_rate,variance,kept\n')
            for c in numeric_cols:
                zr = float(zero_rates.get(c, 0.0))
                var = float(variances.get(c, 0.0))
                kept = 'False' if c in filtered else 'True'
                rf.write(f'{c},{zr},{var},{kept}\n')


        # 9. Create Lagged Features (Crucial for Leakage Prevention)
        logger.info("Creating lagged features for temporal alignment...")
        lag_cols = [c for c in master.columns if any(x in c for x in ['political', 'gdp', 'happiness_score', 'conflict'])]
        for col in lag_cols:
            master[f"{col}_lag1"] = master.groupby('canonical_team')[col].shift(1).fillna(0)

        # Output Master Features
        master.to_csv(self.master_features_path, index=False)
        
        # Save manifest
        manifest = {
            "source_files": ["matches.csv", "political_economic.csv", "conflict_data.csv", "elo_ratings_historical.csv", "culture_happiness.csv"],
            "columns": list(master.columns),
            "lagged_features": [f"{c}_lag1" for c in lag_cols],
            "filtered_features": filtered if 'filtered' in locals() else []
        }
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f)

        # 10. Construct the Match Matrix
        logger.info("Constructing Match matrix with point-in-time features...")
        matches['home_canonical'] = matches['home_team'].apply(standardize_country_name)
        matches['away_canonical'] = matches['away_team'].apply(standardize_country_name)
        
        train_df = matches.copy()
        
        # Features to use: Static + Lagged + Elo (current)
        static_cols = [c for c in master.columns if 'hofstede' in c or 'is_host' in c or 'legacy' in c]
        elo_cols = [c for c in master.columns if 'elo' in c]
        lagged_cols = [f"{c}_lag1" for c in lag_cols]
        perf_cols = [c for c in master.columns if c.startswith('perf_')]
        
        feature_cols = static_cols + elo_cols + lagged_cols + perf_cols
        master_subset = master[['canonical_team', 'year'] + feature_cols]
        
        # Causal Anchor Set
        # Dynamically discover causal anchors from available master columns.
        # Prefer elo, performance, political/economic, happiness, and hofstede-derived columns.
        candidate_patterns = {
            'elo': ['elo_'],
            'perf': ['perf_', 'overall', 'potential'],
            'pol': ['political', 'gdp', 'conflict', 'happiness'],
            'hofstede': ['pdi', 'idv', 'mas', 'uai', 'lto', 'ind']
        }
        discovered = []
        for group, patterns in candidate_patterns.items():
            for pat in patterns:
                col_matches = [c for c in master.columns if pat in c and c not in ['canonical_team', 'year']]
                discovered.extend(col_matches)

        # Deduplicate while preserving order
        seen = set()
        causal_anchors = [x for x in discovered if not (x in seen or seen.add(x))]
        logger.info(f"Using causal anchors discovered from master columns: {causal_anchors}")

        train_df = matches.copy()
        
        # Select subset of master for merging
        master_subset = master[['canonical_team', 'year'] + [c for c in causal_anchors if c in master.columns]]
        
        train_df = train_df.merge(master_subset, left_on=['home_canonical', 'year'], right_on=['canonical_team', 'year'], how='left')
        train_df = train_df.merge(master_subset, left_on=['away_canonical', 'year'], right_on=['canonical_team', 'year'], how='left', suffixes=('_home', '_away'))

        # Calculate differences for causal anchors
        diff_data = {}
        for col in causal_anchors:
            if col in master.columns and col not in ['canonical_team', 'year']:
                h_col, a_col = f"{col}_home", f"{col}_away"
                if h_col in train_df.columns and a_col in train_df.columns:
                    diff_data[f"diff_{col}"] = pd.to_numeric(train_df[h_col], errors='coerce').fillna(0) - pd.to_numeric(train_df[a_col], errors='coerce').fillna(0)
        
        diff_df = pd.DataFrame(diff_data, index=train_df.index)
        y = pd.to_numeric(train_df['home_team_score'], errors='coerce').fillna(0) - pd.to_numeric(train_df['away_team_score'], errors='coerce').fillna(0)
        
        final_train_df = pd.concat([diff_df, y.rename('y')], axis=1)
        final_train_df.to_csv(os.path.join(self.models_dir, 'train.csv'), index=False)
        logger.info(f"Training matrix built with causal anchors. Shape: {final_train_df.shape}")

if __name__ == "__main__":
    converger = FeatureConverger()
    converger.run()
